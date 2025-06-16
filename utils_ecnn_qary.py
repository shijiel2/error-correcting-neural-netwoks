from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os

# Simulating FLAGS using argparse for PyTorch
parser = argparse.ArgumentParser()
parser.add_argument('--indv_CE_lamda', type=float, default=1.0, help="lamda for sum of individual CE")
parser.add_argument('--log_det_lamda', type=float, default=0.5, help="lamda for non-ME")
parser.add_argument('--div_lamda', type=float, default=0.0, help="lamda for non-ME") # Unused in the provided Keras code's custom_loss
parser.add_argument('--augmentation', type=bool, default=True, help="whether use data augmentation")
parser.add_argument('--num_models', type=int, default=30, help="The num of models in the ensemble")
parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
parser.add_argument('--save_dir', type=str, default='/scratch/users/ntu/songy3/codematrix_v2/saved_models/', help="where to save .pt models")
parser.add_argument('--cm_dir', type=str, default='/scratch/users/ntu/songy3/codematrix_v2/saved_codematrix/', help="where to load codematrix")
parser.add_argument('--batch_size', type=int, default=128, help="")
parser.add_argument('--dataset', type=str, default='cifar10', help="mnist or cifar10 or cifar100")

# Parse arguments if the script is run directly, or use defaults
# In a real application, you'd parse args = parser.parse_args()
# For now, we'll create a FLAGS-like object for compatibility with existing code structure.
class Flags:
    def __init__(self):
        default_args = parser.parse_args([]) # Get defaults
        for arg in vars(default_args):
            setattr(self, arg, getattr(default_args, arg))

FLAGS = Flags()


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


def entropy(input_tensor):
    # input_tensor shape is batch_size X num_class
    # Add a small epsilon for numerical stability with log
    return torch.sum(-input_tensor * torch.log(input_tensor + 1e-20), dim=-1)


def ens_div(y_pred, n, q):
    # y_pred are logits before softmax/sigmoid
    div_per_sample = 0
    # Assuming y_pred is already split or handled appropriately by the model's output structure
    # For PyTorch, model output for q-ary would likely be (batch_size, n*q)
    y_pb_logits = torch.split(y_pred, q, dim=-1) # list of n tensors, each (batch_size, q)

    for i in range(n):
        P = F.softmax(y_pb_logits[i], dim=-1)
        div_per_sample += entropy(P) / np.log(q) # Normalize by log(q)
    
    # div_per_sample is currently (batch_size,)
    # Return the mean over the batch
    return torch.mean(div_per_sample / n)


def ce3(y_true_codes, y_pred_logits, n, q):
    # y_true_codes: (batch_size, n), where each element is the true class index (0 to q-1) for that sub-network
    # y_pred_logits: (batch_size, n*q), raw logits from the network
    
    y_true_codes_split = torch.split(y_true_codes, 1, dim=-1) # list of n tensors, each (batch_size, 1)
    y_pred_logits_split = torch.split(y_pred_logits, q, dim=-1) # list of n tensors, each (batch_size, q)
    
    ce = 0
    for i in range(n):
        # y_true_codes_split[i] is (batch_size, 1), containing class indices
        # Squeeze to (batch_size) for CrossEntropyLoss
        # Ensure it's long type for CrossEntropyLoss target
        y_tb_i_indices = y_true_codes_split[i].squeeze(-1).long() 
        
        # CrossEntropyLoss expects logits and class indices
        ce += F.cross_entropy(y_pred_logits_split[i], y_tb_i_indices, reduction='mean')
    return ce / n


def hinge_loss(y_true_codes, y_pred_logits, n, q, cfd_level):
    # y_true_codes: (batch_size, n), where each element is the true class index (0 to q-1)
    # y_pred_logits: (batch_size, n*q), raw logits
    
    y_true_codes_split = torch.split(y_true_codes, 1, dim=-1)
    y_pred_logits_split = torch.split(y_pred_logits, q, dim=-1)
    
    hinge = 0
    for i in range(n):
        y_tb_i_indices = y_true_codes_split[i].squeeze(-1).long() # (batch_size)
        y_pb_i_logits = y_pred_logits_split[i] # (batch_size, q)

        # Create one-hot encoding for true labels to select correct logits
        y_tb_i_one_hot = F.one_hot(y_tb_i_indices, num_classes=q).float() # (batch_size, q)

        correct_logit = torch.sum(y_tb_i_one_hot * y_pb_i_logits, dim=1) # (batch_size)
        
        # For wrong logits, set true class logits to a very small number to not be selected by max
        wrong_logits_mask = (1 - y_tb_i_one_hot) * y_pb_i_logits - y_tb_i_one_hot * 1e4
        wrong_logit = torch.max(wrong_logits_mask, dim=1)[0] # (batch_size)

        hinge_per_subnet = F.relu(wrong_logit - correct_logit + cfd_level)
        hinge += torch.mean(hinge_per_subnet) # Mean over batch for this subnet
    return hinge / n


def custom_loss_fn(n, q, cfd_level, type, current_FLAGS):
    # current_FLAGS to pass the runtime lambda values
    def loss(y_pred_logits, y_true_codes): # y_true_codes are the (0..q-1) indices
        if type == 'ce':
            total_loss = current_FLAGS.indv_CE_lamda * ce3(y_true_codes, y_pred_logits, n, q) + \
                         0.1 * hinge_loss(y_true_codes, y_pred_logits, n, q, cfd_level) - \
                         current_FLAGS.log_det_lamda * ens_div(y_pred_logits, n, q)
        elif type == 'hinge':
            total_loss = current_FLAGS.indv_CE_lamda * hinge_loss(y_true_codes, y_pred_logits, n, q, cfd_level) - \
                         current_FLAGS.log_det_lamda * ens_div(y_pred_logits, n, q)
        else:
            raise ValueError("Unknown loss type")
        return total_loss
    return loss


def ens_div_metric_fn(n, q):
    def ens_div_(y_pred_logits, y_true_codes): # y_true_codes not used but kept for consistent signature
        return ens_div(y_pred_logits, n, q)
    return ens_div_


def ce_metric_fn(n, q): # type argument removed as it's not used in Keras version's logic
    def ce_acc(y_pred_logits, y_true_codes):
        # y_true_codes: (batch_size, n)
        # y_pred_logits: (batch_size, n*q)
        
        y_true_codes_split = torch.split(y_true_codes, 1, dim=-1)
        y_pred_logits_split = torch.split(y_pred_logits, q, dim=-1)
        
        acc = 0
        for i in range(n):
            y_t_i_indices = y_true_codes_split[i].squeeze(-1).long() # (batch_size)
            y_p_i_logits = y_pred_logits_split[i] # (batch_size, q)
            
            # Get predicted class indices
            pred_indices = torch.argmax(F.softmax(y_p_i_logits, dim=1), dim=1)
            
            correct_predictions = (pred_indices == y_t_i_indices).float()
            acc += torch.mean(correct_predictions) # Accuracy for this subnet
            
        return acc / n
    return ce_acc

if __name__ == '__main__':
    # Example usage of FLAGS
    print(f"Number of models from FLAGS: {FLAGS.num_models}")
    FLAGS.num_models = 50 # Can be modified
    print(f"Modified number of models from FLAGS: {FLAGS.num_models}")

    # Example of how you might parse actual command line args
    # import sys
    # actual_args = parser.parse_args(sys.argv[1:])
    # print(f"Actual indv_CE_lamda from command line (if provided): {actual_args.indv_CE_lamda}")