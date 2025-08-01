# drpo_utils.py

from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import os


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    """Initialize tokenizer with proper padding settings."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer


class GPMRewardHead(nn.Module):
    """General Preference Model reward head for non-Bradley-Terry preference models."""
    
    def __init__(self, config: AutoConfig, value_head_dim: int, add_prompt_head: bool = False):
        super().__init__()
        self.value_head = nn.Linear(config.hidden_size, value_head_dim, bias=False)
        self.value_head_dim = value_head_dim
        
        if add_prompt_head:
            # For prompt-dependent preference models
            self.prompt_head = nn.Linear(config.hidden_size, value_head_dim // 2, bias=False)
        else:
            self.prompt_head = None
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reward/preference embedding from hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            rewards: [batch_size, value_head_dim] normalized embeddings
        """
        # Get last non-padded token position for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        
        # Extract hidden states at the last position
        last_hidden = hidden_states[batch_idx, seq_lengths]
        
        # Project to reward space
        rewards = self.value_head(last_hidden)
        
        # Normalize for general preference models
        rewards = F.normalize(rewards, p=2, dim=-1)
        
        return rewards


class GPMwithRewardNetwork(nn.Module):
    def __init__(
        self, 
        model_name_or_path: str,
        device: torch.device = None,
        pad_token_id: Optional[int] = None,
        is_general_preference: bool = True,
        bf16: bool = True
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.is_general_preference = is_general_preference
        
        # Load configuration
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config._attn_implementation = "eager"
        
        # Determine base model class
        base_class = AutoModel._model_mapping[type(config)]
        base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
        
        # Load weights
        try:
            dir_path = snapshot_download(repo_id=model_name_or_path)
        except:
            dir_path = model_name_or_path
            
        combined_weights = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".safetensors"):
                weights = load_file(os.path.join(dir_path, filename))
                combined_weights.update(weights)
        
        # Determine reward head configuration
        value_head_dim = combined_weights.get("value_head.weight", torch.zeros(2, 1)).shape[0]
        add_prompt_head = "prompt_head.weight" in combined_weights
        
        # Create model with custom head
        self.base_model_name = base_class.__name__.lower()
        
        # Initialize base model
        self.model = base_causal_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map=device,
            attn_implementation="eager"
        )
        
        # Add GPM head
        self.gpm_head = GPMRewardHead(config, value_head_dim, add_prompt_head)
        
        # FIX: Load GPM head weights with correct dtype
        dtype = torch.bfloat16 if bf16 else torch.float32
        
        if "value_head.weight" in combined_weights:
            # Convert weight to correct dtype and device
            self.gpm_head.value_head.weight.data = combined_weights["value_head.weight"].to(
                device=device, dtype=dtype
            )
        if add_prompt_head and "prompt_head.weight" in combined_weights:
            self.gpm_head.prompt_head.weight.data = combined_weights["prompt_head.weight"].to(
                device=device, dtype=dtype
            )
        
        # FIX: Ensure the entire GPM head is in the correct dtype
        self.gpm_head = self.gpm_head.to(dtype=dtype)
        
        if pad_token_id is not None:
            self.model.config.pad_token_id = pad_token_id
            
        self.value_head_dim = value_head_dim
        self.to(device)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning preference embeddings."""
        # Ensure EOS token at the end
        input_ids = input_ids.clone()
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            seq_len = attention_mask[i].sum()
            if seq_len > 0:
                input_ids[i, seq_len - 1] = self.model.config.eos_token_id
        
        # Get base model outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Get preference embeddings
        rewards = self.gpm_head(hidden_states, attention_mask)
        
        return rewards


class DPOStyleRewardNetwork(nn.Module):
    """
    A reward network that computes DPO-style rewards.
    
    This network returns rewards of the form:
    r(x, y) = β * log(π(y|x) / π_ref(y|x))
    
    where π is the trained model and π_ref is the reference model.
    This formulation allows using DPO-trained models as reward models
    in a Bradley-Terry preference framework.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        beta: float = 0.1,
        device: Optional[torch.device] = None,
        bf16: bool = True,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the DPO-style reward network.
        
        Args:
            model_name_or_path: Path to the trained HuggingFace model (should be DPO-trained)
            beta: Temperature parameter for the DPO reward scaling
            device: Device to load the model on
            bf16: Whether to use bfloat16 precision
            trust_remote_code: Whether to trust remote code when loading the model
        """
        super().__init__()
        
        self.beta = beta
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Load the trained policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        
        # Ensure model is in eval mode
        self.model.eval()
        
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute log probabilities for the given sequences.
        
        Args:
            model: The model to use for computing log probs
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            temperature: Temperature for scaling logits
            
        Returns:
            Log probabilities summed over the sequence [batch_size]
        """
        with torch.no_grad():
            # Get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            # Get logits and apply temperature
            logits = outputs.logits / (temperature + 1e-7)
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs for actual tokens
            batch_size, seq_len = shift_labels.shape
            gathered_log_probs = log_probs.gather(
                dim=2,
                index=shift_labels.unsqueeze(2)
            ).squeeze(2)
            
            # Mask and sum
            masked_log_probs = gathered_log_probs * shift_mask
            total_log_probs = masked_log_probs.sum(dim=1)
            
            return total_log_probs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_logprobs_sum: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DPO-style rewards.
        
        Args:
            input_ids: Token IDs for the full sequence (prompt + response) [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            ref_log_probs: Precomputed reference model log probabilities [batch_size]
                          If provided, ref_model is ignored
            ref_model: Reference model to compute π_ref(y|x)
                      Required if ref_log_probs is not provided
            temperature: Temperature for scaling logits
            
        Returns:
            rewards: DPO-style rewards β * log(π/π_ref) [batch_size]
            logits: For compatibility with BT models (same as rewards) [batch_size, 1]
        """
        # Compute policy log probabilities
        policy_log_probs = self.compute_log_probs(
            self.model, input_ids, attention_mask, temperature
        )
        
        # Get reference log probabilities
        if ref_logprobs_sum is None:
            if ref_model is None:
                raise ValueError(
                    "Either ref_log_probs or ref_model must be provided"
                )
            ref_logprobs_sum = self.compute_log_probs(
                ref_model, input_ids, attention_mask, temperature
            )
        
        # Compute DPO-style reward: β * log(π/π_ref) = β * (log π - log π_ref)
        rewards = self.beta * (policy_log_probs - ref_logprobs_sum)

        return rewards



def get_preference_score_without_decoding(
    preference_model: nn.Module,
    input_ids_1: torch.Tensor,
    attention_mask_1: torch.Tensor,
    input_ids_2: torch.Tensor,
    attention_mask_2: torch.Tensor,
    is_bt_model: bool = True,
    tokenizer = None,
    prompt_length: Optional[int] = None,
    **kwargs
) -> torch.Tensor:
    """
    Compute preference scores P(response_1 > response_2).
    
    Args:
        preference_model: Either a standard reward model (BT), GPM, or PairRM
        input_ids_1/2: Token IDs for responses [batch_size, seq_len]
        attention_mask_1/2: Attention masks
        is_bt_model: Whether using Bradley-Terry model
        is_pairrm: Whether using PairRM model
        tokenizer: Required for PairRM to decode texts
        prompt_length: Length of prompt tokens (for PairRM)
        
    Returns:
        Preference probabilities [batch_size]
    """
    with torch.no_grad():

        if is_bt_model:
            # Standard Bradley-Terry with scalar rewards
            outputs_1 = preference_model(input_ids=input_ids_1, attention_mask=attention_mask_1)
            outputs_2 = preference_model(input_ids=input_ids_2, attention_mask=attention_mask_2)
            
            rewards_1 = outputs_1.logits.squeeze(-1)
            rewards_2 = outputs_2.logits.squeeze(-1)
            
            # P(y1 > y2) = sigmoid(r1 - r2)
            return torch.sigmoid(rewards_1 - rewards_2)
            
        else:
            # General Preference Model with vector embeddings
            if hasattr(preference_model, 'forward'):
                # Custom GPM model
                embeddings_1 = preference_model(input_ids_1, attention_mask_1)
                embeddings_2 = preference_model(input_ids_2, attention_mask_2)
            else:
                raise ValueError("Unknown preference model type")
            
            # Compute preference using skew-symmetric matrix
            # For 2D: P(y1 > y2) = sigmoid(e1[0]*e2[1] - e1[1]*e2[0])
            if embeddings_1.shape[1] == 2:
                scores = embeddings_1[:, 0] * embeddings_2[:, 1] - embeddings_1[:, 1] * embeddings_2[:, 0]
            else:
                # General case with skew-symmetric matrix
                batch_size = embeddings_1.shape[0]
                dim = embeddings_1.shape[1]
                scores = torch.zeros(batch_size, device=embeddings_1.device)
                
                # Compute e1^T @ R @ e2 where R is skew-symmetric
                for i in range(0, dim, 2):
                    if i + 1 < dim:
                        scores += embeddings_1[:, i] * embeddings_2[:, i + 1]
                        scores -= embeddings_1[:, i + 1] * embeddings_2[:, i]
            
            return torch.sigmoid(scores)