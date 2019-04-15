import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_loss(net, x_input, y_target, device='cpu'):
    net.eval()
    test_batch_size = 1000
    total_loss = 0
    with torch.no_grad():
        for k in range(0, int(y_target.size / test_batch_size) - 1):
            start_ind = k * 1000
            end_ind = (k + 1) * test_batch_size if (k + 1) * test_batch_size < y_target.size else y_target.size
            x = torch.tensor(x_input[start_ind:end_ind, :], device=device, dtype=torch.float)
            y = torch.tensor(y_target[start_ind:end_ind, :], device=device, dtype=torch.float)
            outputs = net(x)
            loss = F.mse_loss(outputs, y)
            total_loss += np.asscalar(loss.cpu().data.numpy())

    return total_loss


def compute_accuracy(net, x_input, y_target, device='cpu'):
    net.eval()
    correct = 0
    total = y_target.size
    test_batch_size = 1000
    with torch.no_grad():
        for k in range(0, int(y_target.size / test_batch_size) - 1):
            start_ind = k * 1000
            end_ind = (k + 1) * test_batch_size if (k + 1) * test_batch_size < y_target.size else y_target.size
            x = torch.tensor(x_input[start_ind:end_ind, :], device=device, dtype=torch.float)
            y = torch.tensor(y_target[start_ind:end_ind, :], device=device, dtype=torch.float)
            outputs = net(x)
            predicted = torch.round(outputs.data)
            correct += (predicted == y).sum().item()
    return correct / total