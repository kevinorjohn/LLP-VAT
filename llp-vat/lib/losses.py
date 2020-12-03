import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.constraints import simplex

from .networks import GaussianNoise


def compute_soft_kl(inputs, targets):
    with torch.no_grad():
        loss = cross_entropy_loss(inputs, targets)
        loss = torch.sum(loss, dim=-1).mean()
    return loss


def compute_hard_l1(inputs, targets, num_classes):
    with torch.no_grad():
        predicted = torch.bincount(inputs.argmax(1),
                                   minlength=num_classes).float()
        predicted = predicted / torch.sum(predicted, dim=0)
        targets = torch.mean(targets, dim=0)
        loss = F.l1_loss(predicted, targets, reduction="sum")
    return loss


def cross_entropy_loss(input, target, eps=1e-8):
    assert simplex.check(input) and simplex.check(target), \
        "input {} and target {} should be a simplex".format(input, target)
    input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input)
    return loss


class ProportionLoss(nn.Module):
    def __init__(self, metric, alpha, eps=1e-8):
        super(ProportionLoss, self).__init__()
        self.metric = metric
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        # input and target shoud ba a probability tensor
        # and have been averaged over bag size
        assert simplex.check(input) and simplex.check(target), \
            "input {} and target {} should be a simplex".format(input, target)
        assert input.shape == target.shape

        if self.metric == "ce":
            loss = cross_entropy_loss(input, target, eps=self.eps)
        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.sum(loss, dim=-1).mean()
        return self.alpha * loss


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        # d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds


class PiModelLoss(nn.Module):
    def __init__(self, std=0.15):
        super(PiModelLoss, self).__init__()
        self.gn = GaussianNoise(std)

    def forward(self, model, x):
        logits1 = model(x)
        probs1 = F.softmax(logits1, dim=1)
        with torch.no_grad():
            logits2 = model(self.gn(x))
            probs2 = F.softmax(logits2, dim=1)
        loss = F.mse_loss(probs1, probs2, reduction="sum") / x.size(0)
        return loss, logits1

