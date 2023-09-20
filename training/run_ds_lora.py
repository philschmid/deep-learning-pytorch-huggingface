from dataclasses import dataclass, field
from typing import cast

import os
import subprocess
from typing import Optional
import torch

from transformers import HfArgumentParser, TrainingArguments, Trainer
from utils.peft_utils import SaveDeepSpeedPeftModelCallback, create_and_prepare_model
from datasets import load_from_disk


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    Additional arguments for training, which are not part of TrainingArguments.
    """
    model_id: str = field(
      metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=False,
    )


def training_function(script_args:ScriptArguments, training_args:TrainingArguments):

    # Load processed dataset from disk
    dataset = load_from_disk(script_args.dataset_path)
    
    # Load and create peft model
    model, peft_config, tokenizer = create_and_prepare_model(script_args.model_id,training_args, script_args)
    model.config.use_cache = False


    # Create trainer and add callbacks
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()
    trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))
    
    # Start training
    trainer.train()

    # Save model on main process
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()

    # TODO: add merge adapters
    # Save everything else on main process
    if trainer.args.process_index == 0:
        if script_args.merge_adapters:
            # merge adapter weights with base model and save
            # save int 4 model
            trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
            # clear memory
            del model
            del trainer
            torch.cuda.empty_cache()

            from peft import AutoPeftModelForCausalLM

            # load PEFT model in fp16
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )  
            # Merge LoRA and base model and save
            model = model.merge_and_unload()        
            model.save_pretrained(
                training_args.output_dir, safe_serialization=True, max_shard_size="8GB"
            )
        else:
            trainer.model.save_pretrained(
                training_args.output_dir, safe_serialization=True
            )

        # save tokenizer 
        tokenizer.save_pretrained(training_args.output_dir)


def main():
    parser = HfArgumentParser([ScriptArguments,TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    script_args = cast(ScriptArguments, script_args)
    training_args = cast(TrainingArguments, training_args)
    
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()