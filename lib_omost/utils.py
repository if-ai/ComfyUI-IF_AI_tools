from contextlib import contextmanager

import torch
import numpy as np


@torch.inference_mode()
def numpy2pytorch(imgs: list[np.ndarray]):
    """Convert a list of numpy images to a pytorch tensor.
    Input: images in list[[H, W, C]] format.
    Output: images in [B, H, W, C] format.

    Note: ComfyUI expects [B, H, W, C] format instead of [B, C, H, W] format.
    """
    assert len(imgs) > 0
    assert all(img.ndim == 3 for img in imgs)
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 255.0
    return h


@contextmanager
def scoped_numpy_random(seed: int):
    state = np.random.get_state()  # Save the current state
    np.random.seed(seed)  # Set the seed
    try:
        yield
    finally:
        np.random.set_state(state)  # Restore the original state


@contextmanager
def scoped_torch_random(seed: int):
    cpu_state = torch.random.get_rng_state()
    gpu_states = []
    if torch.cuda.is_available():
        gpu_states = [
            torch.cuda.get_rng_state(device)
            for device in range(torch.cuda.device_count())
        ]

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if torch.cuda.is_available():
            for idx, state in enumerate(gpu_states):
                torch.cuda.set_rng_state(state, device=idx)
