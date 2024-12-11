import torch
import torch.nn as nn
from torch import Tensor as tensor
import argparse
from typing import Tuple


class FSCLIPmethod(nn.Module):
    '''
    Abstract class for few-shot CLIP methods
    '''
    def __init__(self, args: argparse.Namespace):
        super(FSCLIPmethod, self).__init__()

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                test_features: tensor,
                test_labels: tensor,
                text_weights: tensor,
                clip_model: nn.Module,
                classnames) -> Tuple[tensor, tensor]:

        raise NotImplementedError
