from typing import Dict, Optional, Tuple

import torch

import nanotsparse
from nanotsparse.utils import make_tensor


def build_kmap_implicit_GEMM_hashmap(
    kmap: Dict,
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[Tuple[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
    ifsort: bool = False,
    split_mask_num: int = 1,
    downsample_mode: str = "spconv",
    generative: bool = False,
) -> Dict:
    from nanotsparse.nn import functional as F

    if subm and not generative:
        coords = _coords
    else:
        if not generative:
            coords = F.spdownsample(
                _coords,
                stride,
                kernel_size,
                padding,
                spatial_range,
                downsample_mode=downsample_mode,
            )
        else:
            coords = F.spupsample_generative(_coords, stride, kernel_size, padding, spatial_range)

    kernel_volume = torch.prod(kernel_size)

    to_insert = False
    hashmap = kmap["hashmap"]
    if hashmap is None:
        if coords.device.type == "cpu":
            hashmap = torch.classes.nanotsparse.CPUHashTable(_coords.shape[0])
        else:
            hashmap = torch.classes.nanotsparse.GPUHashTable(_coords.shape[0] * nanotsparse.backends.hash_rsv_ratio)
        to_insert = True

    if to_insert:
        if not generative:
            hashmap.insert_coords(_coords[:, [1, 2, 3, 0]])
        else:
            _insert_coords = _coords.clone()
            _insert_coords[:, 1:] *= stride
            hashmap.insert_coords(_insert_coords[:, [1, 2, 3, 0]])

    if not generative:
        results = (
            hashmap.lookup_coords(
                coords[:, [1, 2, 3, 0]],
                kernel_size.contiguous(),
                stride.contiguous(),
                kernel_volume,
            )
            - 1
        )
    else:
        results = (
            hashmap.lookup_coords(
                coords[:, [1, 2, 3, 0]],
                kernel_size,
                make_tensor((1, 1, 1), dtype=torch.int, device=coords.device),
                kernel_volume,
            )
            - 1
        )

    kmap["out_in_map"] = results
    kmap["coords"] = coords
    kmap["sizes"] = (input_node_num, coords.shape[0])
    kmap["hashmap"] = hashmap

    if ifsort:
        bitmask = torch.ops.nanotsparse.derive_bitmask_from_out_in_map(results, split_mask_num, kmap["sizes"][1])
        sorted_mask, reorder_loc = torch.sort(bitmask, descending=True)
        reorder_loc = reorder_loc.to(torch.int32)
        reorder_out_in_map = torch.ops.nanotsparse.reorder_out_in_map_cuda(results, reorder_loc)
        reduced_sorted_mask = torch.ops.nanotsparse.reduce_bitmask_cuda(sorted_mask, cta_M)
        kmap["reorder_out_in_map"] = reorder_out_in_map
        kmap["reduced_sorted_mask"] = reduced_sorted_mask
        kmap["reorder_loc"] = reorder_loc
        kmap["sorted_mask"] = sorted_mask
    return kmap


def build_kmap_Gather_Scatter_hashmap(
    kmap: Dict,
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[Tuple[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
    downsample_mode: str = "spconv",
    generative: bool = False,
) -> Dict:

    kmap = build_kmap_implicit_GEMM_hashmap(
        kmap,
        input_node_num,
        _coords,
        kernel_size,
        stride,
        padding,
        spatial_range,
        cta_M,
        subm,
        False,
        1,
        downsample_mode,
        generative,
    )

    results = torch.t(kmap["out_in_map"]).contiguous()
    nbsizes = torch.sum(results != -1, dim=1)
    nbmaps = torch.nonzero(results != -1)
    nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]

    nbmaps = nbmaps.contiguous()
    kmap["nbmaps"] = nbmaps
    kmap["nbsizes"] = nbsizes

    return kmap
