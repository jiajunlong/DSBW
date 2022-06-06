import torch
import torch.nn as nn
from torch.nn.functional import conv2d
from model.whitening_iter import whitening_scale_shift


class _DomainSpecificBatchWhitening(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, group_size=4, iters=5, running_m=None,
                 running_inv_sqrt=None, momentum=0.1, track_running_stats=True, eps=1e-5, affine=True):
        super(_DomainSpecificBatchWhitening, self).__init__()
        # self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine,
        # track_running_stats) for _ in range(num_classes)])
        self.bws = nn.ModuleList(
            [whitening_scale_shift(num_features, group_size, iters, running_m, running_inv_sqrt, momentum,
                                   track_running_stats, eps, affine) for _ in range(num_classes)])

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
        bw = self.bws[domain_label[0]]
        return bw(x), domain_label


class DomainSpecificBatchWhitening2d(_DomainSpecificBatchWhitening):
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
