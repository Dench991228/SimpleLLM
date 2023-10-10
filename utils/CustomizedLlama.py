from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from peft import prepare_model_for_kbit_training



def myForward(module: nn.Module):
    # 根据给定的Module返回一个函数，利用Module里面包含的内容进行计算
    def customize_self_attn(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False):
        return module.self_attn(hidden_states,attention_mask,position_ids,past_key_value,output_attentions,use_cache)
    def customize_mlp(x):
        return module.mlp(x)

    def customize_decode(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = module.input_layernorm(hidden_states)
        # Self Attention
        if not module.training:
            hidden_states, self_attn_weights, present_key_value = module.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = torch.utils.checkpoint.checkpoint(
                customize_self_attn,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = module.post_attention_layernorm(hidden_states)
        if not module.training:
            hidden_states = module.mlp(hidden_states)
        else:
            hidden_states = module.mlp(hidden_states)
            #hidden_states = torch.utils.checkpoint.checkpoint(
                #customize_mlp,
                #hidden_states
            #)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    return customize_decode


class CustomizedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for item in self.model.layers:
            # 把各个DecoderLayer里面，只Gradient Checkpoint自注意力机制的部分，不对MLP的部分做Recomputation
            item.forward=myForward(item)
        if hasattr(self, "enable_input_require_grads"):
            self.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.get_input_embeddings().register_forward_hook(make_inputs_require_grad)