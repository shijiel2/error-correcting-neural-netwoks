from __future__ import print_function

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# Note: The model_qary.py conversion is complex. We'll use placeholders or simplified versions.
# Assuming model_qary_pytorch.py contains the PyTorch model definitions
from model_qary import (SharedDense,  # Or SubNetResNetNoFrontShare, etc.
                        SubNetResNet, SubNetResNetWithDecoder)
from utils_ecnn_qary import (FLAGS, ce_metric_fn, custom_loss_fn,
                             ens_div_metric_fn, lr_schedule, ce_pred_fn, code2label)

# from scipy.linalg import hadamard # Not used in this script directly, but was an import

def parse_args():
    parser = argparse.ArgumentParser(description='ECNN')
    # General parameters
    parser.add_argument('--exp_name', type=str, default='exps/test', help='Experiment name')
    parser.add_argument('--num_models', type=int, default=30, help='Number of sub-classifiers (n)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--log_det_lamda', type=float, default=0.1, help='Lambda for log determinant regularization')
    parser.add_argument('--cm_path', type=str, default='./all_matrix/30/2/1.txt', help='Path to the codematrix file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--indv_CE_lamda', type=float, default=5.0, help='Lambda for individual cross-entropy loss')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use (default: cifar10)')
    parser.add_argument('--loss_type', type=str, default='hinge', choices=['ce', 'hinge'], help='Type of loss function to use (default: hinge)')
    parser.add_argument('--hinge_cfd_level', type=int, default=1, help='Hinge CFD level (default: 1)')
    parser.add_argument('--resnet_depth', type=int, default=20, help='Depth of the ResNet model (default: 20)')
    parser.add_argument('--mode', type=str, default='certify', choices=['train', 'test'], help='Mode of operation (default: train)')
    parser.add_argument('--saved_model_path', type=str, default='exps/num_models_30_log_det_lamda_0.1_seed_42_id_1014971/model_epoch_092.pt', help='Path to the saved model for testing (if mode is test)')
    return parser.parse_args()

args = parse_args()

def set_random_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train():
    """Train the model on the CIFAR-10 dataset."""
    def create_coded_labels(original_dataset, code_matrix, num_total_subnets):
        coded_labels = []
        original_images = []
        for i in range(len(original_dataset)):
            image, original_label_idx = original_dataset[i] # original_label_idx is 0-9
            original_images.append(image)
            # code_matrix is (num_classes, num_total_subnets)
            # coded_target_for_sample is (num_total_subnets,)
            coded_target_for_sample = code_matrix[original_label_idx, :]
            coded_labels.append(torch.tensor(coded_target_for_sample, dtype=torch.long)) # long for CrossEntropy with indices, or float for custom
                                                                                        # hinge/ce3 expect indices (long)
        return torch.stack(original_images), torch.stack(coded_labels)

    x_train_tensor, y_train_code_tensor = create_coded_labels(train_dataset_full, cm_numpy, FLAGS.num_models)
    x_test_tensor, y_test_code_tensor = create_coded_labels(test_dataset_full, cm_numpy, FLAGS.num_models)

    train_dataset_coded = TensorDataset(x_train_tensor, y_train_code_tensor)
    test_dataset_coded = TensorDataset(x_test_tensor, y_test_code_tensor)

    train_loader = DataLoader(train_dataset_coded, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset_coded, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    
    # --- Loss Function and Optimizer ---
    criterion = custom_loss_fn(n=FLAGS.num_models, q=qary, cfd_level=hinge_cfd_level, type=loss_type, current_FLAGS=FLAGS)
    optimizer = optim.Adam(model.parameters(), lr=lr_schedule(0)) # Initial LR

    # --- Schedulers ---
    # Keras LearningRateScheduler equivalent
    pt_lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch) / lr_schedule(0) if lr_schedule(0) > 0 else 1.0) # Normalize by initial LR for LambdaLR

    # Keras ReduceLROnPlateau equivalent (optional, Keras script had it but callbacks list only had checkpoint & lr_scheduler)
    # pt_reduce_lr_plateau = ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=5, min_lr=0.5e-6, verbose=True)


    # --- Metrics (PyTorch versions) ---
    # These will be calculated manually in the training/eval loop
    acc_metric_calculator = ce_metric_fn(n=FLAGS.num_models, q=qary)
    div_metric_calculator = ens_div_metric_fn(n=FLAGS.num_models, q=qary)


    # --- Training Loop ---
    logger.info(f"Starting training for {FLAGS.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in tqdm(range(FLAGS.epochs)):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_div = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device) # targets are y_train_code_tensor
            
            optimizer.zero_grad()
            outputs = model(inputs) # Model output should be (batch_size, FLAGS.num_models * q) logits
            
            loss = criterion(outputs, targets) # targets are (batch_size, FLAGS.num_models) indices
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += acc_metric_calculator(outputs.detach(), targets.detach()).item()
            running_div += div_metric_calculator(outputs.detach(), targets.detach()).item() # targets not used by div
            
            if batch_idx % 100 == 99: # logger.info every 100 batches
                logger.info(f"[Epoch {epoch+1}/{FLAGS.epochs}, Batch {batch_idx+1}/{len(train_loader)}] "
                    f"Train Loss: {loss.item():.4f} "
                    f"Train Acc: {acc_metric_calculator(outputs.detach(), targets.detach()).item():.4f} "
                    f"Train Div: {div_metric_calculator(outputs.detach(), targets.detach()).item():.4f}")

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = running_acc / len(train_loader)
        epoch_train_div = running_div / len(train_loader)
        
        logger.info(f"Epoch {epoch+1} Summary: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Train Div: {epoch_train_div:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_div = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_acc += acc_metric_calculator(outputs, targets).item()
                val_div += div_metric_calculator(outputs, targets).item() # targets not used by div

        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_acc = val_acc / len(test_loader)
        epoch_val_div = val_div / len(test_loader)
        
        logger.info(f"Epoch {epoch+1} Summary: Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val Div: {epoch_val_div:.4f}")

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        pt_lr_scheduler.step() # Step LambdaLR based on epoch
        # if using ReduceLROnPlateau: pt_reduce_lr_plateau.step(epoch_val_loss)
        if optimizer.param_groups[0]['lr'] != current_lr:
            logger.info(f"Learning rate updated to: {optimizer.param_groups[0]['lr']}")


        # Save checkpoint (Keras ModelCheckpoint equivalent)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_save_name = model_name_template.format(epoch=epoch + 1)
            model_save_path = os.path.join(save_dir, model_save_name)
            
            # For DataParallel, save module.state_dict()
            model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state_to_save, model_save_path)
            logger.info(f"Epoch {epoch+1}: Validation loss improved to {epoch_val_loss:.4f}, model saved to {model_save_path}")

    logger.info("Finished Training.")

def certify():
    if not args.saved_model_path:
        logger.error("Testing mode requires --saved_model_path to be specified.")
        raise ValueError("For testing mode, --saved_model_path must be provided.")
    
    # train_loader = DataLoader(train_dataset_full, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset_full, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model.load_state_dict(torch.load(args.saved_model_path, map_location=device)) # Load the model state
    eval_model = SubNetResNetWithDecoder(backbone=model, n=FLAGS.num_models, q=qary, cm_numpy=cm_numpy, device=device).to(device) 
    eval_model.eval()

    bin_acc = 0
    log_acc = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred_label_binary = eval_model(inputs, return_type='label_binary')
            pred_label_real = eval_model(inputs, return_type='label_real')

            bin_acc += (pred_label_binary == targets).float().mean().item() # Binary accuracy
            log_acc += (pred_label_real == targets).float().mean().item() #

    logger.info(f"Binary Accuracy: {bin_acc / len(test_loader):.4f}, Logits Accuracy: {log_acc / len(test_loader):.4f}")


if __name__ == "__main__":
    # --- PyTorch Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration (from Keras script, using FLAGS from utils_ecnn_qary) ---
    # These FLAGS would ideally be set by argparse in a main execution block
    FLAGS.augmentation = True
    FLAGS.indv_CE_lamda = args.indv_CE_lamda
    FLAGS.dataset = args.dataset # Keras FLAGS.dataset
    FLAGS.epochs = args.epochs # Keras FLAGS.epochs
    FLAGS.num_models = args.num_models # Keras FLAGS.num_models (number of sub-classifiers, n)
    FLAGS.log_det_lamda = args.log_det_lamda
    FLAGS.batch_size = args.batch_size # Keras FLAGS.batch_size

    set_random_seed(args.seed)

    subtract_pixel_mean = True
    # Dense_code = False # Not directly used in model choice here, but was a flag
    nbits_per_subnet = FLAGS.num_models # Keras: nbits_per_subnet = FLAGS.num_models. This seems to imply n=1 group of subnets.
                                    # If FLAGS.num_models is total subnets, and nbits_per_subnet is how many in one model call,
                                    # then FLAGS.num_models // nbits_per_subnet would be number of SubNetResNet instances.
                                    # The Keras code: if FLAGS.num_models//nbits_per_subnet == 1: model = subnet_resnet(...)
                                    # This implies nbits_per_subnet is the 'n' for a single SubNetResNet call.
                                    # Let's assume FLAGS.num_models is the total number of q-ary classifiers (n_total)
                                    # And nbits_per_subnet is the 'n' parameter for one SubNetResNet instance.
                                    # If they are the same, then one SubNetResNet instance outputs all n_total * q logits.

    loss_type = args.loss_type # 'ce' or 'hinge'
    hinge_cfd_level = args.hinge_cfd_level # Level for hinge loss, typically 1 or 2
    resnet_depth = args.resnet_depth # ResNet depth
    qary = 2      # q-ary classifier
    stack, res_block = 2, 1 # ResNet split point for shared backbone (if used)
    # net_name construction for saving paths
    net_name_suffix = str(qary)+'ary_ens_16x1_'+str(loss_type)+'_resnet'+str(resnet_depth)+'_s'+str(stack)+'r'+str(res_block)+'_Dense32_'+'hinge_cfd_'+str(hinge_cfd_level)+'_set1_nBN' # Keras net_name

    # --- Paths ---
    dir_pwd = os.getcwd()
    dir_name_parts = [
        FLAGS.dataset,
        '_Ensemble_saved_models', str(FLAGS.num_models),
        '_indvCElamda', str(FLAGS.indv_CE_lamda),
        '_logdetlamda', str(FLAGS.log_det_lamda),
        '_submean', str(subtract_pixel_mean),
        # '_dense_', str(Dense_code), # Not used in PyTorch model choice directly
        '_augment_', str(FLAGS.augmentation),
        net_name_suffix
    ]
    dir_name = "".join(dir_name_parts)
    save_dir = os.path.join(dir_pwd, args.exp_name, dir_name)
    # PyTorch model saving typically uses .pt or .pth
    model_name_template = 'model_epoch_{epoch:03d}.pt'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # filepath_template = os.path.join(save_dir, model_name_template) # Used in loop

    # --- Logger Setup ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the lowest-level message to be handled
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(save_dir, 'logging.log'))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # --- Load Codematrix ---
    # cm is num_classes x FLAGS.num_models (n_total)
    # Example: cm for 10 classes, 30 binary classifiers (q=2) would be (10, 30)
    # Each y_train_code[i,j] would be the target (0 or 1) for sample i, classifier j.
    cm_path = args.cm_path # This path seems specific
    try:
        cm_numpy = np.loadtxt(cm_path)
        if cm_numpy.shape[1] != FLAGS.num_models:
            logger.warning(f"Loaded CM shape {cm_numpy.shape} second dim does not match FLAGS.num_models {FLAGS.num_models}. Adjusting FLAGS or CM.")
            # Adjusting FLAGS.num_models based on CM for now, or cm should be num_classes x FLAGS.num_models
            # FLAGS.num_models = cm_numpy.shape[1] # If CM dictates the number of subnets
            # Or, ensure cm_numpy is sliced/processed to be num_classes x FLAGS.num_models
            # Assuming cm_numpy is (num_classes, total_subnets)
            if cm_numpy.shape[0] != 10: # CIFAR-10 has 10 classes
                raise ValueError(f"Code matrix class dimension {cm_numpy.shape[0]} not 10 for CIFAR-10.")
            cm_numpy = cm_numpy[:, :FLAGS.num_models] # Ensure it's (10, FLAGS.num_models)
            logger.info(f"Using CM of shape: {cm_numpy.shape}")

    except IOError:
        logger.info(f"Error: Codematrix file not found at {cm_path}. Using a dummy matrix.")
        num_classes_dataset = 10
        cm_numpy = np.random.randint(0, qary, size=(num_classes_dataset, FLAGS.num_models))


    # --- Data Loading and Preprocessing ---
    input_shape_hw = (32, 32) # CIFAR10 height/width
    input_channels = 3
    cifar10_mean = (0.4914, 0.4822, 0.4465) # Standard CIFAR10 mean
    cifar10_std = (0.2023, 0.1994, 0.2010)  # Standard CIFAR10 std

    transform_list_train = [transforms.ToTensor()]
    transform_list_test = [transforms.ToTensor()]

    if FLAGS.augmentation:
        transform_list_train.extend([
            transforms.RandomCrop(32, padding=4), # Keras width/height_shift_range=0.1, horizontal_flip=True
            transforms.RandomHorizontalFlip(),
        ])

    if subtract_pixel_mean: # Using standard normalization for PyTorch
        transform_list_train.append(transforms.Normalize(cifar10_mean, cifar10_std))
        transform_list_test.append(transforms.Normalize(cifar10_mean, cifar10_std))
    else: # Just scale to [0,1] if not subtracting mean (already done by ToTensor)
        pass 

    train_transform = transforms.Compose(transform_list_train)
    test_transform = transforms.Compose(transform_list_test)

    train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset_full = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # --- Model Definition ---
    # The Keras code structure:
    # if FLAGS.num_models // nbits_per_subnet == 1:
    #     model = subnet_resnet(model_input, nbits_per_subnet, ...)
    # else:
    #     # loop to create multiple subnet_resnet and concatenate outputs
    # This implies nbits_per_subnet is the 'n' for a single SubNetResNet model.
    # And FLAGS.num_models is the total number of q-ary classifiers desired.

    # Let n_for_subnet_module be nbits_per_subnet from Keras.
    # Let n_total_classifiers be FLAGS.num_models from Keras.
    n_for_subnet_module = nbits_per_subnet # Number of q-ary classifiers per SubNetResNet module instance
                                        # In Keras, this was set to FLAGS.num_models, implying one large SubNetResNet
    if n_for_subnet_module != FLAGS.num_models:
        logger.warning("nbits_per_subnet logic from Keras implies it's the 'n' for one SubNetResNet call.")
        logger.info("Setting n_for_subnet_module = FLAGS.num_models for consistency with Keras single model instance.")
        n_for_subnet_module = FLAGS.num_models


    # PyTorch model instantiation
    # Assuming SubNetResNet is designed to output n_for_subnet_module * q logits
    # And shared_dense is for the q-way classification within each of the n_for_subnet_module classifiers.
    pytorch_input_shape = (input_channels, input_shape_hw[0], input_shape_hw[1])

    # The 'shared_dense' in Keras was Dense(q). In PyTorch, this is part of the model.
    # The SubNetResNet in model_qary.py (PyTorch version) takes q_out_features.
    # It also has use_shared_dense_for_subnet_output and shared_dense_out_features (e.g. 32 before q).
    model = SubNetResNet(
        input_shape=pytorch_input_shape,
        n=n_for_subnet_module, # This 'n' is the number of q-ary sub-classifiers in this one model
        depth=resnet_depth,
        dataset=FLAGS.dataset,
        stack_split=stack,      # For shared backbone split point
        res_block_split=res_block, # For shared backbone split point
        q_out_features=qary,       # Each of the 'n' classifiers is q-ary
        shared_dense_out_features=32, # Features before the final q-way dense layer
        use_shared_dense_for_subnet_output=True # If True, SubNetResNet has internal shared q-way dense
    ).to(device)

    # If multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    if args.mode == 'train':
        train()
    elif args.mode == 'certify':
        certify()