import torch
import psutil
import platform
import socket
import multiprocessing
from typing import (
    Any,
    List, 
    Optional
)


def make_list(x: Any) -> list:
    """If a single element into a list of one element.

    Args:
        x (Any): Element(s) to be returned as a list.

    Returns:
        (list): Resulting ``list``.
    """
    return [x] if not isinstance(x, list) and x is not None else x


def get_hardware_specs() -> dict:
    """Returns a ``dict`` with a set of hardware specifications of the host
    computer.

    Returns:
        (dict): Hardware specifications including host name, OS, OS release,
            CPU count, total memory, and GPU.
    """
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


def ops_to_flops(ops_per_frame: int,
                 sample_rate: int,
                 frame_size: int,
                 hop_size: Optional[int] = None) -> float:
    """Calculates the number of floating-point operations per second FLOPs
    based on a give number of operations per frame and frame size.

    !!! note
        Please note that frame size refers to the size of the output of a time
        step in the case of real time models. For example, if a model performs
        an inference over a Short-time Fourier Transform with overlap,
        then the frame size argument of this method would correspond to the
        Short-time Fourier Transform hop size, because that amount represents
        the length of a valid output on each time step.

    Args:
        ops_per_frame (int): Number of operations per frame.
        frame_size (int): Size of the output frame.
        sample_rate (int): Sample rate using with the given frame size.

    Returns:
        (float) Estimate of the floating-point operations per second a model
        performs.
    """
    if sample_rate < frame_size:
        raise ValueError(
            "frame_size should be equal or smaller than sample_rate, otherwise"
            " no inference can be computed per frame"
        )

    if hop_size is None:
        hop_size = frame_size

    inferences_per_second = ((sample_rate - frame_size) / hop_size) + 1.0
    return inferences_per_second * ops_per_frame


def realtime_factor(inference_time: float,
                    sample_rate: int,
                    frame_size: int,
                    hop_size: Optional[int] = None,
                    inverse: bool = False) -> float:
    """Calculates the real-time factor of a model.

    Args:
        inference_time (float): Time in milliseconds needed by a model to
            compute an inference with frame size ``frame_size``, hop size
            ``hop_size`` at a sample rate of ``sample_rate``.
        sample_rate (int): Sample rate used to compute the inference time.
        frame_size (int): Frame size used to compute the inference time.
        hop_size (Optional[int]): Hop size used to compute the inference. If
            ``None`` it will be set to ``frame_size``.
        inverse (bool): If ``True``, the resulting value is calculated as
            ``inference_time * inferences_per_second``, resulting in numbers
            below 1.0 if the model requires less time to compute the inference
            than the time represented by the input samples, otherwise, such
            situation will result in numbers above 1.0.
    """
    if hop_size is None:
        hop_size = frame_size

    inferences_per_second = ((sample_rate - frame_size) / hop_size) + 1.0

    if inverse:
        rtf = (inference_time * inferences_per_second) / 1000.0

    else:
        rtf = 1000.0 / (inferences_per_second * inference_time)

    return rtf


def dict_keys_common(input: dict, query: dict) -> List[str]:
    """Returns a list with all common keys that are in ``query`` ``dict`` and
    in ``input`` ``dict``.

    Args:
        input (dict): Input ``dict``.
        query (dict): Query ``dict``.

    Returns:
        (list): Keys that are present in both ``query`` and ``input``.
    """
    input_keys = list(input.keys())
    common_keys = []

    for k in query:
        if k in input_keys:
            common_keys.append(k)

    return common_keys


def dict_keys_diff(input: dict, query: dict) -> List[str]:
    """Returns a list with all keys that are in ``query`` ``dict`` but not
    in ``input`` ``dict``.

    Args:
        input (dict): Input ``dict``.
        query (dict): Query ``dict``.

    Returns:
        (list): Keys that are present in ``query`` but not in ``input``.
    """
    input_keys = list(input.keys())
    diff_keys = []

    for k in query:
        if k not in input_keys:
            diff_keys.append(k)

    return diff_keys


def dict_merge(input: dict, other: dict) -> dict:
    """Merges two dictionaries (``input`` and ``other``) combining their
    keys and values.

    Args:
        input (dict): Input dictionary.
        other (dict): Dictionary to be merged with ``input``.

    Returns:
        (dict): Merged dictionary.
    """
    # Check all common keys are equal, otherwise merge won't occur
    common_keys = dict_keys_common(input, other)

    for k in common_keys:
        if not input[k] == other[k]:
            raise ValueError(
                "All common keys in 'input' and 'other' dicts should have the "
                "same value, otherwise dicts cannot be merged. Found "
                f"input['{k}']={input[k]} and other['{k}']={other[k]}"
            )

    # Add other keys to input keys
    diff_keys = dict_keys_diff(input, other)

    for k in diff_keys:
        input[k] = other[k]

    return input


def add_extension(file: str, extension: str) -> str:
    """Adds an extension to a filename. If the file already has the extension,
    then the name is returned without modifications.
    
    Args:
        file (str): Input filename.
        extension (str): Extension to be added.
    """
    return f"{file}.{extension}" if not file.endswith(extension) else file
