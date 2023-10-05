import torch
import psutil
import platform
import socket
import multiprocessing
from typing import Any, List


def make_list(x: Any) -> list:
    """ If a single element into a list of one element.

    Args:
        x (Any): Element(s) to be returned as a list.

    Returns:
        (list): Resulting ``list``.
    """
    return [x] if not isinstance(x, list) and x is not None else x


def get_hardware_specs() -> dict:
    gpu_repr = "unavailable"

    if torch.cuda.is_available():
        gpu_repr = []

        num_gpus = torch.cuda.device_count()

        for gpu_idx in range(num_gpus):
            gpu_repr.append(torch.cuda.get_device_name(gpu_idx))

    return {
        "host_name": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu_count": multiprocessing.cpu_count(),
        "total_memory": f"{int(psutil.virtual_memory().total / 1024 ** 2)} MB",
        "gpu": gpu_repr
    }


def ops_to_flops(ops_per_frame: int, frame_size: int, hop_size: int,
                 sample_rate: int) -> float:
    ...


def dict_keys_common(input: dict, query: dict) -> List[str]:
    """ Returns a list with all common keys that are in ``query`` ``dict`` and
    in ``input`` ``dict``.

    Args:
        input (dict): Input ``dict``.
        query (dict): Query ``dict``.

    Returns:
        (list) Keys that are present in both ``query`` and ``input``.
    """
    input_keys = list(input.keys())
    common_keys = []

    for k in query.keys():
        if k in input_keys:
            common_keys.append(k)

    return common_keys


def dict_keys_diff(input: dict, query: dict) -> List[str]:
    """ Returns a list with all keys that are in ``query`` ``dict`` but not
    in ``input`` ``dict``.

    Args:
        input (dict): Input ``dict``.
        query (dict): Query ``dict``.

    Returns:
        (list): Keys that are present in ``query`` but not in ``input``.
    """
    input_keys = list(input.keys())
    diff_keys = []

    for k in query.keys():
        if k not in input_keys:
            diff_keys.append(k)

    return diff_keys


def dict_merge(input: dict, other: dict) -> List[str]:
    # Check all common keys are equal, otherwise merge won't occur
    common_keys = dict_keys_common(input, other)

    for k in common_keys:
        if not input[k] == other[k]:
            raise ValueError(
                "All common keys in input and other dicts should have the same"
                " value, otherwise dicts cannot be merged. Found "
                f"input['{k}']={input[k]} and other['{k}']={other[k]}"
            )

    # Add other keys to input keys
    diff_keys = dict_keys_diff(input, other)

    for k in diff_keys:
        input[k] = other[k]

    return input
