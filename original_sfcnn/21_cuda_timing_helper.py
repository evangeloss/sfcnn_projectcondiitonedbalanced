"""Exported verbatim from notebook code cell 21."""

import time

def sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
