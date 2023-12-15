import torch


def move(m, device):
    if isinstance(m, torch.Tensor):
        m = m.to(device)
    elif isinstance(m, tuple):
        m = tuple(move(m_t, device) for m_t in m)
    elif isinstance(m, list):
        m = [move(m_t, device) for m_t in m]
    elif isinstance(m, dict):
        m = {k: move(v, device) for k, v in m.items()}
    else:
        raise Exception(f"m should be either tensor,tuple,list or dict,got{type(m)}")

    return m
