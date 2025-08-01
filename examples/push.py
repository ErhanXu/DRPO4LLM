from transformers import AutoModelForCausalLM, AutoTokenizer

output_dir = "./drdrpo-pythia-1b-tldr"
model_id = "Eehan/pythia-1b-drpo-lora-tldr-665"

model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)