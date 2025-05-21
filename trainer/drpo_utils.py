from typing import Optional, List, Dict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, pipeline,AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import AutoTokenizer     
import os
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from datasets import load_dataset
import re
from dataclasses import dataclass, field
from typing import Optional
from datasets import  DatasetDict
from datasets import load_dataset
from huggingface_hub import ModelCard
from trl.trainer.utils import selective_log_softmax


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer

def get_reward_model(base_causal_model, base_llm_model, value_head_dim: int, add_prompt_head: bool, is_general_preference: bool=False):
    class CustomRewardModel(base_causal_model):

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            self.is_general_preference = is_general_preference   
            
            self.value_head = nn.Linear(config.hidden_size, value_head_dim, bias=False) 
            if add_prompt_head:
                self.prompt_head = nn.Linear(config.hidden_size, value_head_dim // 2, bias=False)

        def custom_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            
            if not self.is_general_preference:
                values = self.value_head(last_hidden_states).squeeze(-1)
                # left padding in training mode
                if self.training:
                    reward = values[:, -1]
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    reward = values.gather(dim=1, index=eos_indices).squeeze(1)
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
            else:
                values = self.value_head(last_hidden_states)
                # left padding in training mode
                if self.training:
                    reward = values[:, -1, :]
                    reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)
                    eos_indices = eos_indices.unsqueeze(1)  # Change shape to [batch_size, 1]                  
                    reward_list = []
                    for dim in range(self.value_head.out_features):
                        reward_list.append(values[:,:,dim].gather(dim=1, index=eos_indices))
                    reward = torch.cat(reward_list, dim=1)
                    reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
        
        def create_skew_symmetric_block_matrix(self, dim, device, dtype, prompt_hidden_states):
            """
            Create a batch of skew-symmetric block matrices where each matrix is data-dependent on
            the corresponding prompt_hidden_states. Only the relevant block diagonal parts are generated.
            
            Args:
            - dim: Dimension of the square matrix (must be even).
            - prompt_hidden_states: Tensor of shape [batch_size, hidden_dim].
            
            Returns:
            - batch_R_matrices: Tensor of shape [batch_size, dim, dim], with skew-symmetric block entries.
            """
            if hasattr(self, 'prompt_head'):
                batch_size = prompt_hidden_states.shape[0]
                
                # Ensure that dim is even, as we're creating blocks of size 2x2
                assert dim % 2 == 0, "dim must be even for skew-symmetric block generation"

                # Pass through the linear layer to get the block diagonal entries (half of the matrix's off-diagonal blocks)
                block_values = self.prompt_head(prompt_hidden_states).view(batch_size, dim // 2)
                block_values = torch.softmax(block_values, dim=-1)
                
                # Create a batch of zero matrices [batch_size, dim, dim]
                batch_R_matrices = torch.zeros((batch_size, dim, dim), device=device, dtype=dtype)
                
                # Fill only the block diagonal entries with the learned values
                for i in range(0, dim, 2):
                    batch_R_matrices[:, i, i + 1] = -block_values[:, i // 2]
                    batch_R_matrices[:, i + 1, i] = block_values[:, i // 2]  # Skew-symmetric condition
            else:
                raise AttributeError("prompt_head is not defined. Ensure 'add_prompt_head' is set to True during initialization.")
                
            return batch_R_matrices
         
    return CustomRewardModel

class GPMwithRewardNetwork(nn.Module):
    def __init__(self, 
                 model_name_or_path: str, 
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
                 pad_token_id: Optional[int]=None,
                 is_general_preference: bool=True,
                 bf16: bool=True):
        super().__init__()
        self.device = device

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config._attn_implementation = "eager"
        base_class = AutoModel._model_mapping[type(config)]
        base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
        try:
            dir_path = snapshot_download(repo_id=model_name_or_path)
        except Exception as e:
            dir_path = model_name_or_path
        combined_weights = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".safetensors"):
                file_path = os.path.join(dir_path, filename)
                weights = load_file(file_path)
                combined_weights.update(weights)

        if "value_head.weight" in combined_weights:
            self.value_head_dim = combined_weights["value_head.weight"].shape[0]
        self.add_prompt_head = True if "prompt_head.weight" in combined_weights else False 

        cls_class = get_reward_model(base_causal_class, base_class, add_prompt_head=self.add_prompt_head, value_head_dim=self.value_head_dim, is_general_preference=is_general_preference)
        self.model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map="auto",
            attn_implementation="eager"
        )

        if pad_token_id is not None:
            self.rm.config.pad_token_id = pad_token_id
    
    def forward(self, input_ids, attention_mask, return_output=False):
        input_ids[:, -1] = self.model.config.eos_token_id
        attention_mask[:, -1] = 1
        with torch.no_grad():
            rewards, _ = self.model.custom_forward(input_ids, attention_mask, return_output)
        return rewards


class GPMPipeline:
    def __init__(self, model_name_or_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), is_general_preference: bool=True, bf16: bool=True, truncation: bool=True, max_length: int=4096, padding: bool=True, tau: float=0.1):
        self.device = device
        self.is_general_preference = is_general_preference

        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.tau = tau
        
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config._attn_implementation = "eager" 
        base_class = AutoModel._model_mapping[type(config)]
        base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)

        try:
            dir_path = snapshot_download(repo_id=model_name_or_path)
        except Exception as e:
            dir_path = model_name_or_path
        combined_weights = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".safetensors"):
                file_path = os.path.join(dir_path, filename)
                weights = load_file(file_path)
                combined_weights.update(weights)
        if "value_head.weight" in combined_weights:
            self.value_head_dim = combined_weights["value_head.weight"].shape[0]



        self.add_prompt_head = True if "prompt_head.weight" in combined_weights else False

        cls_class = get_reward_model(base_causal_class, base_class, add_prompt_head=self.add_prompt_head, value_head_dim=self.value_head_dim, is_general_preference=is_general_preference)

        # configure model
        self.model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map="auto",
            attn_implementation="eager"
        )
        
        # configure tokenizer
        self.tokenizer = get_tokenizer(model_name_or_path, self.model, "left", use_fast=True)
        self.tokenizer.truncation_side = "right"
        
        # prepare model
        self.model.to(device)
        self.model.eval()

    def __call__(self, samples: List[str], return_prompt=False):
        input_texts = samples
        inputs = self.tokenizer(
            input_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        inputs["input_ids"][:, -1] = self.tokenizer.eos_token_id
        inputs["attention_mask"][:, -1] = 1

        with torch.no_grad():
            rewards, _ = self.model.custom_forward(**inputs, return_output=return_prompt)

        return rewards
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        return self




def get_preference_score(preference_model, a_1_iuput, a_2_input, is_bt_model:bool = True, kwargs: Optional[Dict] = None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), noisy=0.0):
    if kwargs.get("indifferent", False):
        return 0.5*torch.ones(len(a_1_iuput)).to(device)
    if kwargs.get("random", False):
        return torch.rand(len(a_1_iuput)).to(device)
    a1_reward = preference_model(a_1_iuput)
    a2_reward = preference_model(a_2_input)

    if is_bt_model:
        result = a1_reward - a2_reward
    else:
        if preference_model.value_head_dim == 2:
            result = a1_reward[:, 0] * a2_reward[:, 1] - a1_reward[:, 1] * a2_reward[:, 0]
        else:
            R_matrix = torch.zeros((preference_model.value_head_dim, preference_model.value_head_dim), device=a1_reward.device, dtype=a1_reward.dtype)
            for i in range(0, preference_model.value_head_dim, 2):
                R_matrix[i, i+1] = -1 
                R_matrix[i+1, i] = 1   
            if a1_reward.device == a2_reward.device == R_matrix.device:
                transformed_a_1 = torch.matmul(a1_reward, R_matrix.T)
                result = torch.bmm(transformed_a_1.view(a1_reward.shape[0], 1, preference_model.value_head_dim), a2_reward.view(a2_reward.shape[0], preference_model.value_head_dim, 1))
                result = result.view(a1_reward.shape[0])  
    p = F.sigmoid(result) + noisy * (torch.randn_like(result))
    if kwargs.get("reverse", False):
        p = 1 - p
    return p


class BTPipeline:
    def __init__(self, model_name_or_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), truncation: bool=True, padding: bool=True):
        self.pipeline = pipeline("sentiment-analysis", model=model_name_or_path, device=device)
        self.truncation=truncation
        self.padding=padding

    def __call__(self, input_text: List[str]):
        batch_size = len(input_text)
        sentiment_results = self.pipeline(input_text, batch_size=batch_size, truncation=self.truncation, padding=self.padding)
        return torch.tensor([res["score"] if res["label"] == "POSITIVE" else 1 - res["score"] for res in sentiment_results]).to(self.device)
        
    def to(self, device):
        self.device = device
        self.pipeline.model.to(device)
        return self 

class BTwithRewardPipeline:
    def __init__(self, reward_model_id, model_name_or_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), truncation: bool=False, padding: bool=True):
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            reward_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.rm_tokenizer.pad_token is None:
            self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token
            self.rm_tokenizer.pad_token_id = self.rm_tokenizer.eos_token_id
            self.rm.config.pad_token_id = self.rm_tokenizer.pad_token_id
            self.rm.config.padding_side = "right"
        
        self.device = device
        self.truncation = truncation
        self.padding = padding

    def __call__(self, input_text: List[str]):
        if isinstance(input_text, str):
            input_text = [input_text]
        
        inputs = self.rm_tokenizer(
            input_text,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True
        )

        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.rm(**inputs)
            scores = output.logits.squeeze()
            # Handle both single and batch inputs
            if scores.ndim == 0:
                return scores.unsqueeze(0)
            return scores
        
    def to(self, device):
        self.device = device
        self.rm.to(device)
        return self     

class estDPOStylePipeline:
    def __init__(self, model_name_or_path: dict, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), truncation: bool=True, padding: bool=True, beta: float=0.05, max_length: int=512):
        """
        calculate preference by DPO-style estimation,
        given prompt_response, policy_model and the reference_model,
        this will return beta*(log p(prompt_response) - log p_ref(prompt_response))
        no worries about counting in the probs of prompt, 
        since it will be cancelled out when computing the sigmoid(a1_reward - a2_reward), where a1_reward is just self.__call__(a1)
        which is beta*(log p(prompt_a1) - log p_ref(prompt_a1))
        note model_name_or_path should be a dict with keys 'policy' and 'reference'
        """
        self.device = device
        self.beta = beta
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

        self.policy = AutoModelForCausalLM.from_pretrained(model_name_or_path["policy"]).to(device)
        self.reference = AutoModelForCausalLM.from_pretrained(model_name_or_path["reference"]).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path["policy"])
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, input_text: List[str]) -> torch.Tensor:
        try:
            policy_logprobs = self._calc_logprobs(self.policy, input_text)
            reference_logprobs = self._calc_logprobs(self.reference, input_text)
            diff = policy_logprobs - reference_logprobs
            result = self.beta * diff
        finally:
            torch.cuda.empty_cache()
        return result

    def _calc_logprobs(self, model, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_attention_mask=True
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            input_ids = inputs["input_ids"][:, 1:]
            attention_mask = inputs["attention_mask"][:, 1:]
            logprobs = selective_log_softmax(logits, input_ids)
            final_score = (logprobs * attention_mask).sum(-1)
        del inputs, outputs, logits, input_ids, attention_mask, logprobs
        torch.cuda.empty_cache()
        return final_score

    def to(self, device: torch.device):
        self.device = device
        self.policy.to(device)
        self.reference.to(device)
        return self


def get_preference_score_without_decoding(preference_model, a1_iuput_ids, a1_attention_mask, a2_input_ids, a2_attention_mask, is_bt_model:bool = True, kwargs: Optional[Dict] = None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    if kwargs.get("indifferent", False):
        return 0.5*torch.ones(len(a1_iuput_ids)).to(device)
    if kwargs.get("random", False):
        return torch.rand(len(a1_iuput_ids)).to(device)
    a1_reward = preference_model(a1_iuput_ids, a1_attention_mask)
    a2_reward = preference_model(a2_input_ids, a2_attention_mask)
    if is_bt_model:
        result = a1_reward - a2_reward
    else:
        if preference_model.value_head_dim == 2:
            result = a1_reward[:, 0] * a2_reward[:, 1] - a1_reward[:, 1] * a2_reward[:, 0]
        else:
            R_matrix = torch.zeros((preference_model.value_head_dim, preference_model.value_head_dim), device=a1_reward.device, dtype=a1_reward.dtype)
            for i in range(0, preference_model.value_head_dim, 2):
                R_matrix[i, i+1] = -1 
                R_matrix[i+1, i] = 1   
            if a1_reward.device == a2_reward.device == R_matrix.device:
                transformed_a_1 = torch.matmul(a1_reward, R_matrix.T)
                result = torch.bmm(transformed_a_1.view(a1_reward.shape[0], 1, preference_model.value_head_dim), a2_reward.view(a2_reward.shape[0], preference_model.value_head_dim, 1))
                result = result.view(a1_reward.shape[0])  
    p = F.sigmoid(result)
    if kwargs.get("reverse", False):
        p = 1 - p
    return p



class BTRewardNetwork(nn.Module):
    def __init__(self, 
    model_name_or_path: str, 
    revision: str = "main", 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
    pad_token_id: Optional[int]=None, 
    eos_token_id: Optional[int]=151645,
    ):
        super().__init__()
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation="flash_attention_2",
            num_labels=1,
        )
        if pad_token_id is not None:
            self.rm.config.pad_token_id = pad_token_id
        self.to(device)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.rm(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs.logits.squeeze()
            if scores.ndim == 0:
                scores = scores.unsqueeze(0)
            return scores
        
    def to(self, device):
        self.device = device
        self.rm.to(device)
        return self
    
