from typing import List, Optional, Tuple

import torch
import transformers
from peft.tuners.lora import LoraLayer

try:
    from flash_attn import flash_attn_func
except Exception:
    raise ModuleNotFoundError(
        "Please install FlashAttention first, e.g., with pip install flash-attn --no-build-isolation, Learn more at https://github.com/Dao-AILab/flash-attention#installation-and-features"
    )

try:
    from einops import rearrange
except Exception:
    raise ModuleNotFoundError("Please install einops first, e.g., with pip install einops")


# ADAPTED https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/falcon_flash_attn_monkey_patch.py
def forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
    num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, query_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(
        batch_size * num_kv_heads,
        query_length,
        self.head_dim,
    )
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

    past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
    query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length)

    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, kv_length, head_dim]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)

    _, kv_length, _ = key_layer.shape
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None
    attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float("-1e9")).to(query_layer.dtype)
    query_layer_ = (
        query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim).transpose(1, 2).to(torch.bfloat16)
    )
    key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim).transpose(1, 2).to(torch.bfloat16)
    value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim).transpose(1, 2).to(torch.bfloat16)

    if alibi is not None:
        raise ValueError("`alibi` is not supported when `use_flash_attn` is True")

    # below output will have shape (batch_size, seqlen, nheads, headdim)
    attn_output = flash_attn_func(query_layer_, key_layer_, value_layer_, causal=True)
    attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
    output_tensor = self.dense(attn_output)
    return output_tensor, present


def replace_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        print(
            "Flash attention is only supported on Ampere or Hopper GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.falcon.modeling_falcon.FalconAttention.forward = forward
    

def unplace_flash_attn_with_attn():
    import importlib
    import transformers

    print("Reloading falcon model, unpatching flash attention")
    importlib.reload(transformers.models.falcon.modeling_falcon)


# Adapted from https://github.com/tmm1/axolotl/blob/2eda9e02a9d15a7a3f92b41f257d9844d72fc220/src/axolotl/utils/models.py#L338
def upcast_layer_for_flash_attention(model, torch_dtype):
    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(torch_dtype)
        if "norm" in name:
            module.to(torch_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)

    return model
