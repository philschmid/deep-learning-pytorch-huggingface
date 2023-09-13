from dataclasses import dataclass, field
from typing import cast

import os
import subprocess
from typing import Optional

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
    # Load and create peft model
    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    model.config.use_cache = False

    # Load processed dataset from disk
    dataset = load_from_disk(script_args.dataset_path)

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

    # Save everything else on main process
    if trainer.args.process_index == 0:
        print("Sharding model if >10GB...")
        # FSDP/DeepSpeed save the model as a single `pytorch_model.bin` file, so we need to shard it.
        # We run this in a subprocess to avoid interference from the accelerators.
        subprocess.run(
            [
                "python",
                "shard_checkpoint.py",
                f"--output_dir={training_args.output_dir}",
            ],
            check=True,
        )
        if "training_args.bin" in os.listdir(training_args.output_dir):
            os.remove(os.path.join(training_args.output_dir, "training_args.bin"))
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