from dataclasses import dataclass, field
import tempfile
from typing import Optional
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import HfApi

# Example usage:
# python scripts/merge_adapter_weights.py --peft_model_id falcon-180b-lora-fa --output_dir merged-weights --save_tokenizer True

def save_model(model_path_or_id, save_dir, save_tokenizer=True):
  model = AutoPeftModelForCausalLM.from_pretrained(
      model_path_or_id,
      low_cpu_mem_usage=True,
      torch_dtype=torch.float16,
  )  
  # Merge LoRA and base model and save
  model = model.merge_and_unload()        
  model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="3GB")
  
  # save tokenizer
  if save_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
    tokenizer.save_pretrained(save_dir) 


@dataclass
class ScriptArguments:
    peft_model_id: str = field(metadata={"help": "model id or path to model"})
    output_dir: Optional[str] = field(default="merged-weights", metadata={"help": "where the merged model should be saved"})
    save_tokenizer: Optional[bool] = field(default=True, metadata={"help": "whether to save the tokenizer"})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "whether to push the model to the hub"})
    repository_id: Optional[str] = field(default=None, metadata={"help": "the model name"})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]
api = HfApi()

if args.push_to_hub:
  repo_id = args.repository_id if args.repository_id else args.peft_model_id.split('/')[-1]
  with tempfile.TemporaryDirectory() as temp_dir:
    save_model(args.peft_model_id, temp_dir, args.save_tokenizer)
    api.upload_large_folder(
      folder_path=temp_dir,
      repo_id=repo_id,
      repo_type="model",
    )
else:
  save_model(args.peft_model_id, args.output_dir, args.save_tokenizer)