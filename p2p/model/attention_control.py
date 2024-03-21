import torch
from typing import Optional

from model.ptp_utils import LocalBlend
from model import seq_aligner
from model.attention_base import AttentionControlEdit

class AttentionReplace(AttentionControlEdit):

    def __init__(self, prompts,tokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,device = torch.device("cuda:0"),LOW_RESOURCE=False,dtype=torch.float32):
        super(AttentionReplace, self).__init__(prompts, tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend,device,LOW_RESOURCE)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device).to(dtype)
    
    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
        

class AttentionRefine(AttentionControlEdit):

    def __init__(self, prompts, tokenizer,num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,device = torch.device("cuda:0"),LOW_RESOURCE=False):
        super(AttentionRefine, self).__init__(prompts, tokenizer,num_steps, cross_replace_steps, self_replace_steps, local_blend,device,LOW_RESOURCE)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace


class AttentionReweight(AttentionControlEdit):

    def __init__(self, prompts, tokenizer,num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None,device = torch.device("cuda:0"),LOW_RESOURCE=False,dtype=torch.float32):
        super(AttentionReweight, self).__init__(prompts,tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend,device, LOW_RESOURCE)
        self.equalizer = equalizer.to(device).to(dtype)
        self.prev_controller = controller

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace