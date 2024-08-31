import torch
from lightning.fabric.utilities import measure_flops

from src.models.wide_resnet import wide_resnet


def format_number(n):
    power = 1000
    cnt = 0
    power_units = {0: '', 1: 'K', 2: 'M', 3: 'B', 4: 'T', 5: 'P'}
    while n > power:
        n /= power
        cnt += 1
    return f'{n:.3f} {power_units[cnt]}'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    with torch.device("meta"):
        model = wide_resnet(depth=28, width=2, num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 32, 32)

    model_fwd = lambda: model(x)
    fwd_flops = measure_flops(model, model_fwd)

    print(format_number(fwd_flops))
    print(format_number(count_parameters(model)))
