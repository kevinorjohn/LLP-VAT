def accuracy(output, target, top_k=(1, )):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


def parameters_string(module):
    lines = [
        "",
        "Model name: {}".format(module.__class__.__name__),
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(
            row_format.format(name=name,
                              shape=" * ".join(str(p) for p in param.size()),
                              total_size=param.numel()))
    lines.append("=" * 75)
    lines.append(
        row_format.format(name="all parameters",
                          shape="sum of above",
                          total_size=sum(
                              int(param.numel()) for name, param in params)))
    lines.append("")
    return "\n".join(lines)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {
            name + postfix: meter.val
            for name, meter in self.meters.items()
        }

    def averages(self, postfix='/avg'):
        return {
            name + postfix: meter.avg
            for name, meter in self.meters.items()
        }

    def sums(self, postfix='/sum'):
        return {
            name + postfix: meter.sum
            for name, meter in self.meters.items()
        }

    def counts(self, postfix='/count'):
        return {
            name + postfix: meter.count
            for name, meter in self.meters.items()
        }


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format)
