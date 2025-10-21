from src.frame_fetchers.abstract import AbstractFrameFetcher
from src.frame_fetchers.opencv import OpencvFrameFetcher

# NvDec is optional - requires PyNvCodec which may not be available
try:
    from src.frame_fetchers.nvdec import NvDecFrameFetcher
except ImportError:
    NvDecFrameFetcher = None
