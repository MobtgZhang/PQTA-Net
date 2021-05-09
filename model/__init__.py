from .model import DocReader
from .rnet import RNet
from .ptqnet import PQTANet
from .mreader import MReader

from .bert import ErniePQTANet,ErnieRNet,ErnieMReader
__all__ = [
    'RNet',
    'PQTANet',
    'MReader',
    'ErniePQTANet',
    'ErnieRNet',
    'DocReader',
    'ErnieMReader'
]
