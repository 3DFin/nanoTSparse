import torch

import torchsparse.backend

__all__ = ["convert_transposed_out_in_map"]

def convert_transposed_out_in_map(out_in_map, size):
    out_in_map_t = torch.full(
        (size, out_in_map.shape[1]),
        fill_value=-1,
        device=out_in_map.device,
        dtype=torch.int32,
    )
    torchsparse.backend.convert_transposed_out_in_map(out_in_map, out_in_map_t)
    return out_in_map_t
