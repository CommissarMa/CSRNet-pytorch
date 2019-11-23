import numpy as np
import torch


def denormalize(tensor):
    mean = [0.5, 0.5, 0.5]
    std = [0.225,0.225,0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor