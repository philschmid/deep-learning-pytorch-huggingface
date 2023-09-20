import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    TrainingArguments,
)
from utils.falcon_patch import replace_attn_with_flash_attn as replace_falcon_attn_with_flash_attn
from utils.llama_patch import replace_attn_with_flash_attn as replace_llama_attn_with_flash_attn


class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control



def create_and_prepare_model(model_id:str, training_args:TrainingArguments, script_args):
    if script_args.use_flash_attn:
        replace_falcon_attn_with_flash_attn()
        replace_llama_attn_with_flash_attn()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_cache=not training_args.gradient_checkpointing,
    )
    print("model loaded")

    # find all linear modules in model for lora
    target_modules = find_all_linear_names(model)

    # create lora config
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    # enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    # pre-process the model by upcasting the layer norms in float 32 for
    # Adapted from https://github.com/tmm1/axolotl/blob/2eda9e02a9d15a7a3f92b41f257d9844d72fc220/src/axolotl/utils/models.py#L338
    print("pre-processing model for peft")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.bfloat16)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                module = module.to(torch.bfloat16)

    # initialize peft model
    print("initializing peft model")
    model = get_peft_model(model, peft_config)

    # logger.info parameters
    model.print_trainable_parameters()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)