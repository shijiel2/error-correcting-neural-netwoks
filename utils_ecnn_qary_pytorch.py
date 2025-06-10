import torch
import torch.nn.functional as F
from argparse import Namespace

FLAGS = Namespace(
    indv_CE_lamda=1.0,
    log_det_lamda=0.5,
    div_lamda=0.0,
    augmentation=True,
    num_models=30,
    epochs=200,
    save_dir='/tmp/',
    cm_dir='/tmp/',
    batch_size=128,
    dataset='cifar10'
)

def lr_schedule(epoch: int) -> float:
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate:', lr)
    return lr

def entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x * (x.clamp(min=1e-20)).log()).sum(dim=-1)

def ens_div(y_true: torch.Tensor, y_pred: torch.Tensor, n: int, q: int) -> torch.Tensor:
    div = 0
    y_pb = torch.split(y_pred, q, dim=-1)
    for i in range(n):
        P = F.softmax(y_pb[i], dim=-1)
        div = div + entropy(P) / torch.log(torch.tensor(float(q)))
    return div / n

def ce3(y_true: torch.Tensor, y_pred: torch.Tensor, n: int, q: int) -> torch.Tensor:
    ce = 0
    y_tb = torch.split(y_true, 1, dim=-1)
    y_pb = torch.split(y_pred, q, dim=-1)
    for i in range(n):
        y_tb_i = F.one_hot(y_tb[i].long().squeeze(-1), num_classes=q).float()
        ce = ce + F.cross_entropy(y_pb[i], y_tb_i, reduction='none')
    return ce / n

def hinge_loss(y_true: torch.Tensor, y_pred: torch.Tensor, n: int, q: int, cfd_level: float) -> torch.Tensor:
    hinge = 0
    y_tb = torch.split(y_true, 1, dim=-1)
    y_pb = torch.split(y_pred, q, dim=-1)
    for i in range(n):
        y_tb_i = F.one_hot(y_tb[i].long().squeeze(-1), num_classes=q).float()
        correct_logit = (y_tb_i * y_pb[i]).sum(dim=1)
        wrong_logits = (1 - y_tb_i) * y_pb[i] - y_tb_i * 1e4
        wrong_logit = wrong_logits.max(dim=1).values
        hinge = hinge + F.relu(wrong_logit - correct_logit + cfd_level)
    return hinge / n

def custom_loss(n: int, q: int, cfd_level: float, loss_type: str):
    def loss(y_true, y_pred):
        if loss_type == 'ce':
            total_loss = FLAGS.indv_CE_lamda * ce3(y_true, y_pred, n, q) + 0.1 * hinge_loss(y_true, y_pred, n, q, cfd_level) - FLAGS.log_det_lamda * ens_div(y_true, y_pred, n, q)
        else:
            total_loss = FLAGS.indv_CE_lamda * hinge_loss(y_true, y_pred, n, q, cfd_level) - FLAGS.log_det_lamda * ens_div(y_true, y_pred, n, q)
        return total_loss.mean()
    return loss

def ens_div_metric(n: int, q: int):
    def metric(y_true, y_pred):
        return ens_div(y_true, y_pred, n, q).mean()
    return metric

def ce_metric(n: int, q: int):
    def metric(y_true, y_pred):
        y_t = torch.split(y_true, 1, dim=-1)
        y_p = torch.split(y_pred, q, dim=-1)
        acc = 0
        for i in range(n):
            y_t_i = F.one_hot(y_t[i].long().squeeze(-1), num_classes=q).float()
            pred = F.softmax(y_p[i], dim=-1)
            acc = acc + (pred.argmax(dim=1) == y_t_i.argmax(dim=1)).float()
        return acc / n
    return metric
