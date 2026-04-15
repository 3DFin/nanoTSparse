from typing import Dict, Tuple, Optional
import torch

import torchsparse.backend
import torchsparse.backends
from torchsparse.utils import make_tensor


def build_kmap_implicit_GEMM_hashmap_on_the_fly(
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
) -> Dict:
    kmap["coords"] = _coords
    kmap["spatial_range"] = spatial_range
    # coords = _coords[:, [3, 0, 1, 2]]

    coords = _coords.contiguous()
    if spatial_range is not None:
        coords_max_tuple = tuple(x - 1 for x in spatial_range)
        coords_max = make_tensor(
            coords_max_tuple, dtype=torch.int, device=coords.device
        )
    else:
        coords_max = coords.max(0).values
        if not subm:
            coords_max[1:] = (
                coords_max[1:] + 2 * padding - (kernel_size - 1)
            ) // stride

    if torchsparse.tensor.get_allow_negative_coordinates():
        coords_min = coords.min(0).values
        coords_min[1:] = torch.div(
            coords_min[1:] - 2 * padding + (kernel_size - 1), stride
        )
    else:
        coords_min = make_tensor((0, 0, 0, 0), dtype=torch.int, device=coords.device)

    if subm:
        hash_func = torch.ops.nanots.build_kernel_map_subm_hashmap
    else:
        hash_func = torch.ops.nanots.build_kernel_map_downsample_hashmap


    assert (
        torchsparse.backends.hash_rsv_ratio >= 2
    ), f"hash_rsv_ratio should be no less than 2, now {torchsparse.backends.hash_rsv_ratio}."

    to_insert = False
    hashmap = kmap["hashmap"]
    if hashmap is None:
        hashmap = torch.classes.nanots.GPUHashTable(_coords.shape[0] * torchsparse.backends.hash_rsv_ratio)
        to_insert = True

    out = hash_func(
        hashmap,
        coords,
        coords_min,
        coords_max,
        kernel_size,
        stride,
        padding,
        to_insert,
    )

    # update kernel_map
    out_in_map = out[0]
    kmap["out_in_map"] = out_in_map
    kmap["hashmap"] = hashmap
    
    if len(out) != 1:
        coords = out[1]
        # coords = coords[:, [1, 2, 3, 0]]
        kmap["coords"] = coords
    kmap["sizes"] = (input_node_num, coords.shape[0])

    if ifsort:
        bitmask = torch.ops.nanots.derive_bitmask_from_out_in_map(
            out_in_map, split_mask_num, kmap["sizes"][1]
        )
        sorted_mask, reorder_loc = torch.sort(bitmask, descending=True)
        reorder_loc = reorder_loc.to(torch.int32)
        reorder_out_in_map = torch.ops.nanots.reorder_out_in_map_cuda(
            out_in_map, reorder_loc
        )
        reduced_sorted_mask = torch.ops.nanots.reduce_bitmask_cuda(
            sorted_mask, cta_M
        )
        kmap["reorder_out_in_map"] = reorder_out_in_map
        kmap["reduced_sorted_mask"] = reduced_sorted_mask
        kmap["reorder_loc"] = reorder_loc
        kmap["sorted_mask"] = sorted_mask

    return kmap


def build_kmap_Gather_Scatter_hashmap_on_the_fly(
    kmap: Dict,
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[Tuple[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
) -> Dict:

    kmap = build_kmap_implicit_GEMM_hashmap_on_the_fly(
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
    )

    results = torch.t(kmap["out_in_map"]).contiguous()
    nbsizes = torch.sum(results != -1, dim=1)
    nbmaps = torch.nonzero(results != -1)
    nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]
    # important for build masks
    nbmaps = nbmaps.contiguous()

    # compute mask for GPU implementation
    # it's only available when using conv_mode > 0
    input_mask, output_mask = torch.ops.nanots.build_mask_from_kmap(
        _coords.shape[0],
        kmap["coords"].shape[0],
        nbmaps.int(),
        nbsizes.int()[0 : kmap["coords"].shape[0]],
    )

    kmap["nbmaps"] = nbmaps
    kmap["nbsizes"] = nbsizes
    kmap["input_mask"] = input_mask
    kmap["output_mask"] = output_mask

    return kmap
