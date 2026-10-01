"""Row 152 (MODEL-23): a strands-style pointer added to decider's one-slot readout. strands-decider (REPORT 146) scores option k from
the hidden state at the last token of option k's line against the state at the answer position, through a small head trained from
scratch; decider reads the option's label token from the LM head at "Category: (". The hybrid adds the two:
    logit_k = <h_slot, W_lab[label_k]>  +  gate * <q(LN(h_slot)), k(LN(h_line_k))> / sqrt(dim)
with the options listed again after the query (oneslot.build_layout relist=True) so each line's last token has read the history. The
gate starts at 1: a zero gate would also hold the pointer's q and k at zero gradient until it moved (the r152 smoke ended at 0.000 after
three warmup steps), and at initialisation the pointer's logits are small (std ~0.3) against the label logits (~10), so the start is
still essentially plain decider. The head is fp32 (~1.3M
parameters at 2B, dim 256), saved next to the LoRA as pointer_head.pt.
"""
import torch
from torch import nn

FILE = "pointer_head.pt"


class OptionPointer(nn.Module):
    def __init__(self, hidden, dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, dim)
        self.k = nn.Linear(hidden, dim)
        self.gate = nn.Parameter(torch.ones(()))
        self.scale = dim ** -0.5

    def forward(self, slot, lines):
        """slot [d], lines [K, d] (any dtype) -> [K] fp32 logits to add to the label logits."""
        q = self.q(self.norm(slot.float()))
        k = self.k(self.norm(lines.float()))
        return self.gate * (k @ q) * self.scale
