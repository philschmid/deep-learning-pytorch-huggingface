from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, HfArgumentParser

# execute script
# python scripts/merge_peft_into_model.py --peft_model_id falcon-180b-lora-fa --output_dir merged-weights --save_tokenizer True

@dataclass
class ScriptArguments:
    peft_model_id: str = field(metadata={"help": "model id or path to model"})
    output_dir: Optional[str] = field(default="merged-weights", metadata={"help": "where the merged model should be saved"})
    save_tokenizer: Optional[bool] = field(default=True, metadata={"help": "whether to save the tokenizer"})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "whether to push the model to the hub"})
    repository_id: Optional[str] = field(default=None, metadata={"help": "the model name"})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

model = AutoPeftModelForCausalLM.from_pretrained(
    args.peft_model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)  
# Merge LoRA and base model and save
model = model.merge_and_unload()        
model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="4GB")

if args.save_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.peft_model_id)
    tokenizer.save_pretrained(args.output_dir)
    
if args.push_to_hub:
  if args.repository_id is None:
    raise ValueError("You must specify a repository id to push to the hub")
  from huggingface_hub import HfApi
  api = HfApi()
  api.upload_folder(
    folder_path=args.output_dir,
    repo_id=args.repository_id,
    repo_type="model",
  )