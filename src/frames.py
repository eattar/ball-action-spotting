import abc
from typing import Type

import torch


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    frames = frames.to(torch.float32) / 255.0
    return frames


def pad_to_frames(frames: torch.Tensor,
                  size: tuple[int, int],
                  pad_mode: str = "constant",
                  fill_value: int = 0) -> torch.Tensor:
    height, width = frames.shape[-2:]
    target_width, target_height = size
    
    # If frames are larger than target size, resize first
    if height > target_height or width > target_width:
        # Resize maintaining aspect ratio to fit within target size
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Convert to float for interpolation if needed
        original_dtype = frames.dtype
        if frames.dtype == torch.uint8:
            frames = frames.float()
        
        # Resize using interpolate
        frames = torch.nn.functional.interpolate(
            frames,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert back to original dtype
        if original_dtype == torch.uint8:
            frames = frames.to(torch.uint8)
        
        height, width = new_height, new_width
    
    height_pad = target_height - height
    width_pad = target_width - width
    assert height_pad >= 0 and width_pad >= 0

    top_height_pad: int = height_pad // 2
    bottom_height_pad: int = height_pad - top_height_pad
    left_width_pad: int = width_pad // 2
    right_width_pad: int = width_pad - left_width_pad
    frames = torch.nn.functional.pad(
        frames,
        [left_width_pad, right_width_pad, top_height_pad, bottom_height_pad],
        mode=pad_mode,
        value=fill_value,
    )
    return frames


class FramesProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        pass


class PadNormalizeFramesProcessor(FramesProcessor):
    def __init__(self,
                 size: tuple[int, int],
                 pad_mode: str = "constant",
                 fill_value: int = 0):
        self.size = size
        self.pad_mode = pad_mode
        self.fill_value = fill_value

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        frames = pad_to_frames(frames, self.size,
                               pad_mode=self.pad_mode,
                               fill_value=self.fill_value)
        frames = normalize_frames(frames)
        return frames


_FRAME_PROCESSOR_REGISTRY: dict[str, Type[FramesProcessor]] = dict(
    pad_normalize=PadNormalizeFramesProcessor,
)


def get_frames_processor(name: str, processor_params: dict) -> FramesProcessor:
    assert name in _FRAME_PROCESSOR_REGISTRY
    return _FRAME_PROCESSOR_REGISTRY[name](**processor_params)
