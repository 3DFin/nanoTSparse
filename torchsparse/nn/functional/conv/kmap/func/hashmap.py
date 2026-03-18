from mpmath import ker
from typing import Dict, Tuple, Union, Optional
import torch

import torchsparse.backend
from torchsparse.utils import make_tensor


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
    from torchsparse.nn import functional as F

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
            ) # create Fout, for stride = 1, Fin == Fout
        else:
            coords = F.spupsample_generative(
                _coords, stride, kernel_size, padding, spatial_range
            )

    kernel_volume = torch.prod(kernel_size)

    to_insert = False
    if kmap["hashmap_keys"] is None:
        kmap["hashmap_keys"] = torch.zeros(
            2 * _coords.shape[0], dtype=torch.int64, device=coords.device
        )
        to_insert = True
    if kmap["hashmap_vals"] is None:
        kmap["hashmap_vals"] = torch.zeros(
            2 * _coords.shape[0], dtype=torch.int32, device=coords.device
        )
    hashmap = torchsparse.backend.CPUHashTable()

    print( "TO INSERT", to_insert)
    if to_insert:
        if not generative:
            hashmap.insert_coords(_coords[:, [1, 2, 3, 0]].to(torch.int64))
        else:
            _insert_coords = _coords.clone()
            _insert_coords[:, 1:] *= stride
            hashmap.insert_coords(_insert_coords[:, [1, 2, 3, 0]].to(torch.int64)
)

    if not generative:
        results = (
            hashmap.lookup_coords(
                coords[:, [1, 2, 3, 0]].to(torch.int64)
, kernel_size, stride, kernel_volume
            )
            - 1 # create indices # -1 shift make 0 based indexing, since 0 has a special meaning in hashmap implementation
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

    if ifsort:
        bitmask = torchsparse.backend.derive_bitmask_from_out_in_map(
            results, split_mask_num, kmap["sizes"][1]
        )
        sorted_mask, reorder_loc = torch.sort(bitmask, descending=True)
        reorder_loc = reorder_loc.to(torch.int32)
        reorder_out_in_map = torchsparse.backend.reorder_out_in_map_cuda(
            results, reorder_loc
        )
        reduced_sorted_mask = torchsparse.backend.reduce_bitmask_cuda(
            sorted_mask, cta_M
        )
        kmap["reorder_out_in_map"] = reorder_out_in_map
        kmap["reduced_sorted_mask"] = reduced_sorted_mask
        kmap["reorder_loc"] = reorder_loc
        kmap["sorted_mask"] = sorted_mask

    return kmap

#mask negative offsets
def build_mask_from_kmap_cpu(n_points, n_out_points, kmap, kmap_sizes):
   #print(kmap_sizes)
    kernel_volume = kmap_sizes.shape[0]
    print(kernel_volume)

    input_mask = torch.full((kernel_volume*n_points,), -1, dtype=torch.int)
    output_mask = torch.full((kernel_volume*n_out_points,), -1, dtype=torch.int)

    offset = 0 # start of the definition of the current k
    for k in range(kernel_volume):
        n_neighbors = kmap_sizes[k].item() #num of offsets with this K
        if n_points == n_out_points and kernel_volume % 2 == 1 and (k == (int(kernel_volume) // int(2))):
            print("central k size:", kernel_volume)
            offset += n_neighbors;
            continue
        for i in range(n_neighbors):
            neighbor_idx = kmap[offset + i]
            input_mask[k*n_points + neighbor_idx[0]] =  i
            output_mask[k*n_out_points + neighbor_idx[1]] = i
        offset += n_neighbors
    return input_mask, output_mask


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

    #print("kmap_shape", kmap["out_in_map"].shape)
    #print(kmap["out_in_map"])
    results = torch.t(kmap["out_in_map"]).contiguous()
    #print("result", results.shape)
    #print(results)
    nbsizes = torch.sum(results != -1, dim=1)
    nbmaps = torch.nonzero(results != -1)
    #print("NBmap shape:", nbmaps.shape)
    #print(nbmaps)
    nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]
    #print("NBmap shape 2:", nbmaps.shape)
    #print(nbmaps)

    # important for build masks
    nbmaps = nbmaps.contiguous()
    input_mask, output_mask = build_mask_from_kmap_cpu(
        _coords.shape[0],
        kmap["coords"].shape[0],
        nbmaps.int(),
        nbsizes.int(),
    )

    kmap["nbmaps"] = nbmaps
    kmap["nbsizes"] = nbsizes
    kmap["input_mask"] = input_mask
    kmap["output_mask"] = output_mask
    #print(output_mask)
    return kmap


def build_kmap_Fetch_on_Demand_hashmap(
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
    nbsizes = torch.sum(results != -1, dim=1).to(torch.int)
    nbmaps = torch.nonzero(results != -1)
    nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]

    kernel_volume = nbsizes.size(0)
    nbaddrs = torch.zeros((kernel_volume + 1), dtype=torch.int, device=nbmaps.device)
    qnbaddrs = torch.zeros((kernel_volume + 1), dtype=torch.int, device=nbmaps.device)

    # Derive quantified arrays
    torchsparse.backend.exclusive_scan_quantified_wrapper(
        kernel_volume, nbsizes, nbaddrs, qnbaddrs
    )

    # nbmaps need to be transposed for Fetch-on-Demand
    kmap["nbmaps"] = nbmaps.transpose(0, 1).int()
    kmap["nbsizes"] = nbsizes

    kmap["nbaddrs"] = nbaddrs
    kmap["qnbaddrs"] = qnbaddrs
    kmap["qmapsize"] = qnbaddrs[-1].cpu().int()

    return kmap
