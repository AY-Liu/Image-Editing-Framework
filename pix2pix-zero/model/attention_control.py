import torch
from diffusers.models.attention_processor import Attention

class MyAttnProcessor():
    def __call__(self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # save the attention_map in attn_probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet(unet):
    original_processors = {}
    # set the gradients for XA maps to be true
    for name, params in unet.named_parameters():
        if 'attn2' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention":
            original_processors[name] = module.get_processor()
            module.set_processor(MyAttnProcessor())
    return unet,original_processors

def restore_original_processors(unet, original_processors):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and name in original_processors:
            module.set_processor(original_processors[name])
