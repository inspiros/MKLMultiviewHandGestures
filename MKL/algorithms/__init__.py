from .base import MKL, AverageMKL
from .komd import KOMD
from .EasyMKL import EasyMKL
from .SimpleMKL import SimpleMKL
#from HeuristicMKLClassifier import HeuristicMKLClassifier

__all__ = ['EasyMKL',
           'KOMD',
           'MKL',
           'AverageMKL',
           'SimpleMKL', # (Not working properly)
           ]
