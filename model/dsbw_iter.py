import torch
import torch.nn as nn
from torch.nn.functional import conv2d
from model.iterative_normalization_FlexGroup import IterNorm
from model.group_whitening import GroupItN

class _DomainSpecificBatchWhitening(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, num_channels=32, T=5,
        momentum=0.1, eps=1e-5, affine=True, whitening="BW"):
        super(_DomainSpecificBatchWhitening, self).__init__()
        # self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine,
        # track_running_stats) for _ in range(num_classes)])
        if whitening == "GW":
            self.bns = nn.ModuleList(
                [GroupItN(num_features=num_features, num_groups=num_channels, T=T, eps=eps, momentum=momentum,
                                        affine=affine) for _ in range(num_classes)])
        else:
            self.bns = nn.ModuleList(
                [IterNorm(num_features=num_features, num_channels=num_channels, T=T, eps=eps, momentum=momentum,
                                        affine=affine) for _ in range(num_classes)])

    # def reset_running_stats(self):
    #     for bw in self.bws:
    #         bw.reset_running_stats()

    # def reset_parameters(self):
    #     for bw in self.bws:
    #         bw.reset_parameters()

    def _check_input_dim(self, x):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class DomainSpecificBatchWhitening2d(_DomainSpecificBatchWhitening):
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
