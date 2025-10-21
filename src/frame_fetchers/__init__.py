from src.frame_fetchers.abstract import AbstractFrameFetcher
from src.frame_fetchers.opencv import OpencvFrameFetcher

# Try to import NvDec, but make it optional
try:
    from src.frame_fetchers.nvdec import NvDecFrameFetcher
except ImportError:
    # PyNvCodec not available, NvDecFrameFetcher won't be available
    NvDecFrameFetcher = None
