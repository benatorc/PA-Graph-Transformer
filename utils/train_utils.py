import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
import pdb


def get_grad_norm(model, debug=False):
    total_norm = 0
    for name, param in model.named_parameters():
        if param is not None and param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if debug:
                pdb.set_trace()
    total_norm = total_norm ** (1. / 2)
    return total_norm


def backprop_grads(model, optimizer, stats_tracker, args):
    pre_clip_norm = get_grad_norm(model)
    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    post_clip_norm = get_grad_norm(model, debug=False)
    stats_tracker.add_stat('gnorm', pre_clip_norm, 1)
    stats_tracker.add_stat('gnorm_clip', post_clip_norm, 1)

    optimizer.step()


def compute_acc(input_probs, target, n_classes=1):
    if n_classes > 1:
        preds = np.argmax(input_probs, axis=1)
        acc = np.mean(preds == target)
    else:
        preds = (input_probs > 0.5).astype(int)
        acc = np.mean(preds == target)
    return acc


def compute_auc(input_probs, target):
    auc = metrics.roc_auc_score(y_true=target, y_score=input_probs)
    return auc
