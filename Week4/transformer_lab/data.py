import torch


PAD_ID = 0
BOS_ID = 1


def make_batch(batch_size, seq_len, vocab_size, task, device):
    """Generate a synthetic sequence transduction batch.

    The generated tokens are in [1, vocab_size). PAD_ID is reserved for future
    extension but is not used by the fixed-length tasks.
    """
    x = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    if task == "copy":
        y = x.clone()
    elif task == "reverse":
        y = torch.flip(x, dims=[1])
    elif task == "sort":
        y = torch.sort(x, dim=1).values
    else:
        raise ValueError(f"unknown task: {task}")
    return x, y


def make_decoder_input(y):
    bos = torch.full((y.size(0), 1), BOS_ID, dtype=y.dtype, device=y.device)
    return torch.cat([bos, y[:, :-1]], dim=1)
