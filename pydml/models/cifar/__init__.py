"""CIFAR models module."""

from pydml.models.cifar.resnet import resnet32, resnet110
from pydml.models.cifar.mobilenet import mobilenet_v2
from pydml.models.cifar.wrn import wrn_28_10

__all__ = ["resnet32", "resnet110", "mobilenet_v2", "wrn_28_10"]
