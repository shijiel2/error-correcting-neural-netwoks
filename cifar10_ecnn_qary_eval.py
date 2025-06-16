from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Assuming utils_ecnn_qary_pytorch.py and model_qary_pytorch.py are available
from utils_ecnn_qary import FLAGS # Using the FLAGS object from utils
from model_qary import SubNetResNet, SharedDense, decoder as model_decoder # Or other models as needed

# CleverHans - PyTorch integration can be tricky.
# For now, we'll focus on model loading and basic evaluation.
# Adversarial attack generation would require a PyTorch-compatible library or manual implementation.
# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
# from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
# (These are examples, actual CleverHans PyTorch API might differ or need specific wrappers)

# --- PyTorch Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Configuration (from Keras script) ---
# These FLAGS are expected to be populated by utils_ecnn_qary
FLAGS.dataset = 'cifar10'
FLAGS.augmentation = True # Affects data loading if it were training, less so for eval pre-saved model
subtract_pixel_mean = True
# FLAGS.num_models = 30 # This will be set per test case from t_cnnX
# Dense_code = False # Not directly used in model choice here

# Attack parameters (Keras version)
# att_para_keras = (['FGSM', [0.0, 0.04]],
#                   ['PGD', [0.04]])
# For PyTorch, attack parameters might need adjustment.
# For now, we'll focus on clean accuracy.
att_para_pytorch = (['Clean', [0.0]],) # Placeholder for clean evaluation

# Model configurations to test (example from Keras t_cnn6)
# t_cnn_config_name, num_subnets (n_total), best_epoch, indv_lamda, div_lamda, q_value
t_cnn6_configs = [
    ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1', 10, 151, 5.0, 0.5, 2],
    ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1', 10, 150, 5.0, 0.0, 2],
    # ... add other configs from t_cnn6 if needed
]

# --- Fixed Parameters from Keras script ---
depth_val = 20
stack_val, res_block_val = 2, 1 # Split point for shared backbone in SubNetResNet
nbits_per_subnet_val = 10 # This was 'nbits_per_subnet' in Keras.
                          # If FLAGS.num_models // nbits_per_subnet == 1, then one SubNetResNet.
                          # This implies nbits_per_subnet is the 'n' for one SubNetResNet module.

# --- Data Loading (CIFAR-10) ---
input_shape_hw = (32, 32)
input_channels = 3
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

transform_list_test = [transforms.ToTensor()]
if subtract_pixel_mean:
    transform_list_test.append(transforms.Normalize(cifar10_mean, cifar10_std))
test_transform = transforms.Compose(transform_list_test)

# Load CIFAR-10 test set, get original class labels (0-9)
test_dataset_original_labels = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
# We also need the y_test as one-hot for final accuracy calculation against model's class predictions
y_test_original_indices = np.array(test_dataset_original_labels.targets)
y_test_one_hot_numpy = np.eye(10)[y_test_original_indices]
y_test_one_hot_tensor = torch.tensor(y_test_one_hot_numpy, dtype=torch.float32).to(device)

test_loader_original_labels = DataLoader(test_dataset_original_labels, batch_size=FLAGS.batch_size, shuffle=False, num_workers=2)


# --- Main Evaluation Loop ---
if __name__ == "__main__":
    # Select which set of configurations to run
    test_configs_to_run = t_cnn6_configs
    test_opts_indices = [-2] # Keras: test_opts = [-2], picks second to last from t_cnnX

    for att_config_idx in range(len(att_para_pytorch)):
        att_name = att_para_pytorch[att_config_idx][0]
        gammas = att_para_pytorch[att_config_idx][1] # Attack strengths (0.0 for clean)

        for model_config_idx_in_list in test_opts_indices:
            # Resolve the actual index from the list of configs
            if model_config_idx_in_list < 0:
                actual_model_config_idx = len(test_configs_to_run) + model_config_idx_in_list
            else:
                actual_model_config_idx = model_config_idx_in_list
            
            if not (0 <= actual_model_config_idx < len(test_configs_to_run)):
                print(f"Skipping invalid model_config_idx: {model_config_idx_in_list}")
                continue

            current_config = test_configs_to_run[actual_model_config_idx]
            net_name_suffix_eval = current_config[0]
            FLAGS.num_models = current_config[1] # Total number of q-ary sub-classifiers (n_total)
            n_epoch_load = current_config[2]
            FLAGS.indv_CE_lamda = current_config[3] # Not used for eval, but part of path
            FLAGS.log_det_lamda = current_config[4] # Not used for eval, but part of path
            q_val = current_config[5]

            print(f"\n--- Evaluating Model: {net_name_suffix_eval} (Epoch: {n_epoch_load}, Q: {q_val}, N_total: {FLAGS.num_models}) ---")
            print(f"Attack: {att_name}")

            # Construct model path (consistent with cifar10_ecnn_qary_v5.py saving)
            dir_pwd_eval = '/home/twp/work/songy/codematrix_v5_qary'
            dir_name_parts_eval = [
                FLAGS.dataset,
                '_Ensemble_saved_models', str(FLAGS.num_models),
                '_indvCElamda', str(FLAGS.indv_CE_lamda),
                '_logdetlamda', str(FLAGS.log_det_lamda),
                '_submean', str(subtract_pixel_mean),
                # '_dense_', str(Dense_code), # Was in Keras path
                '_augment_', str(FLAGS.augmentation), # Augmentation flag was part of training path
                net_name_suffix_eval # This should match the suffix from training
            ]
            save_dir_eval = os.path.join(dir_pwd_eval, "".join(dir_name_parts_eval))
            model_filename_to_load = f'model_epoch_{n_epoch_load:03d}.pt' # Matches v5 saving
            model_filepath_to_load = os.path.join(save_dir_eval, model_filename_to_load)

            if not os.path.exists(model_filepath_to_load):
                print(f"Model file not found: {model_filepath_to_load}. Skipping.")
                continue
            
            # Load Code Matrix (cm0 from Keras eval script)
            # This cm0 is (num_classes, n_total_classifiers) with values 0..q-1
            # The Keras eval script then converts this to one-hot and concatenates for the 'linear' decoder.
            # cm_for_decoder will be (num_classes, n_total_classifiers * q) if one-hot encoded.
            # Or, if the decoder handles 0..q-1 internally, it's (num_classes, n_total_classifiers)
            cm0_path = f'/home/twp/work/songy/all_matrix/Q/{FLAGS.num_models}/RC/{q_val}/1.txt'
            try:
                cm0_numpy = np.loadtxt(cm0_path) # Shape: (num_classes, FLAGS.num_models)
                if cm0_numpy.shape != (10, FLAGS.num_models): # CIFAR-10
                    raise ValueError(f"Loaded cm0 shape {cm0_numpy.shape} mismatch expected (10, {FLAGS.num_models})")
            except Exception as e:
                print(f"Error loading cm0 from {cm0_path}: {e}. Skipping this config.")
                continue

            # For the 'linear' decoder in PyTorch (LinearDecoder class in model_qary.py)
            # It expects cm_numpy to be (num_classes, num_input_features_to_decoder)
            # The input to this decoder is the output of the main model (sigmoid applied to logits).
            # Main model output: (batch, FLAGS.num_models * q) logits. After sigmoid: (batch, FLAGS.num_models * q) probabilities.
            # So, LinearDecoder's cm_numpy should be (num_classes, FLAGS.num_models * q)
            # The Keras code:
            # cm1 = []
            # for n_idx in range(FLAGS.num_models): # Iterate over each of the N sub-classifiers
            #     cm1.append(tf.keras.utils.to_categorical(cm0_numpy[:, n_idx], q_val)) # cm0[:,n] is (num_classes,), to_cat gives (num_classes, q)
            # cm_for_keras_decoder = np.concatenate(cm1, axis=-1) # Results in (num_classes, FLAGS.num_models * q)
            
            cm1_list = []
            for n_idx in range(FLAGS.num_models):
                one_hot_col = np.eye(q_val)[cm0_numpy[:, n_idx].astype(int)] # (num_classes, q)
                cm1_list.append(one_hot_col)
            cm_for_pytorch_decoder_numpy = np.concatenate(cm1_list, axis=-1) # (num_classes, FLAGS.num_models * q)


            # --- Model Definition and Loading ---
            # Determine n_for_subnet_module based on Keras logic
            # Keras: if FLAGS.num_models // nbits_per_subnet_val == 1: ...
            # This implies nbits_per_subnet_val is the 'n' for one SubNetResNet module.
            # And FLAGS.num_models is the total number of q-ary classifiers.
            n_for_subnet_module_eval = nbits_per_subnet_val
            if FLAGS.num_models != n_for_subnet_module_eval:
                 print(f"Warning: Eval script's FLAGS.num_models ({FLAGS.num_models}) "
                       f"differs from nbits_per_subnet_val ({n_for_subnet_module_eval}). "
                       f"Assuming SubNetResNet was trained with n={n_for_subnet_module_eval} subnets internally if that was fixed, "
                       f"or n={FLAGS.num_models} if nbits_per_subnet_val was set to FLAGS.num_models during training.")
                 # For consistency with v5 training script, n_for_subnet_module was FLAGS.num_models
                 n_for_subnet_module_eval = FLAGS.num_models


            base_model = SubNetResNet(
                input_shape=(input_channels, input_shape_hw[0], input_shape_hw[1]),
                n=n_for_subnet_module_eval, # Number of q-ary classifiers in this one model
                depth=depth_val,
                dataset=FLAGS.dataset,
                stack_split=stack_val,
                res_block_split=res_block_val,
                q_out_features=q_val,
                shared_dense_out_features=32, # Assuming 32 from net_name_suffix_eval
                use_shared_dense_for_subnet_output=True # Matches v5 training
            ).to(device)

            # The Keras eval script then appends a final decoder
            # model_output = decoder(model_output, opt='linear', cm=cm_for_keras_decoder)
            # In PyTorch, we wrap the base_model and the new decoder layer.
            
            # The input features to the 'linear' decoder are FLAGS.num_models * q_val (after sigmoid)
            # The cm_for_pytorch_decoder_numpy is (num_classes, FLAGS.num_models * q_val)
            final_decoder_layer = model_decoder(
                opt='linear', 
                cm_numpy=cm_for_pytorch_decoder_numpy # Expected by LinearDecoder in model_qary
            ).to(device)

            # Combine base_model and final_decoder_layer
            class EnsembleModelWithFinalDecoder(nn.Module):
                def __init__(self, backbone, final_dec):
                    super().__init__()
                    self.backbone = backbone
                    self.final_dec = final_dec
                
                def forward(self, x_input):
                    # Backbone output is (batch, n_total_classifiers * q_val) logits
                    backbone_logits = self.backbone(x_input)
                    # The LinearDecoder in model_qary applies sigmoid internally to these logits
                    final_class_scores = self.final_dec(backbone_logits) # (batch, num_classes)
                    return final_class_scores # These are scores, not necessarily probabilities yet.
                                              # Keras LinearDecoder had a relu. For classification, usually softmax.
                                              # For eval, argmax on these scores is fine.

            eval_model = EnsembleModelWithFinalDecoder(base_model, final_decoder_layer).to(device)
            
            if torch.cuda.device_count() > 1:
                eval_model = nn.DataParallel(eval_model)
            
            try:
                eval_model.load_state_dict(torch.load(model_filepath_to_load, map_location=device))
                # If saved DataParallel model.module.state_dict(), and loading to single GPU or different DataParallel:
                # state_dict = torch.load(model_filepath_to_load, map_location=device)
                # from collections import OrderedDict
                # new_state_dict = OrderedDict()
                # for k, v in state_dict.items():
                #     name = k[7:] if k.startswith('module.') else k # remove `module.`
                #     new_state_dict[name] = v
                # if isinstance(eval_model, nn.DataParallel): eval_model.module.load_state_dict(new_state_dict)
                # else: eval_model.load_state_dict(new_state_dict)
                print(f"Successfully loaded model weights from {model_filepath_to_load}")
            except Exception as e:
                print(f"Error loading model weights: {e}. Check saved model structure (DataParallel etc.). Skipping.")
                continue
                
            eval_model.eval()

            for gamma_val in gammas: # Loop over attack strengths
                print(f"  Gamma (attack strength): {gamma_val}")
                
                correct_predictions = 0
                total_samples = 0
                
                # Adversarial attack generation would go here if gamma_val > 0
                # For now, only clean evaluation (gamma_val = 0.0)

                with torch.no_grad():
                    for batch_idx, (inputs, true_class_indices) in enumerate(test_loader_original_labels):
                        inputs, true_class_indices = inputs.to(device), true_class_indices.to(device)
                        
                        # If adversarial attack:
                        # if gamma_val > 0 and att_name == 'FGSM':
                        #    inputs_adv = fast_gradient_method(eval_model, inputs, eps=gamma_val, norm=np.inf, targeted=False) # Example
                        # elif gamma_val > 0 and att_name == 'PGD':
                        #    inputs_adv = projected_gradient_descent(eval_model, inputs, eps=gamma_val, eps_iter=0.01, nb_iter=10, norm=np.inf) # Example
                        # else:
                        inputs_adv = inputs # Clean input

                        outputs_class_scores = eval_model(inputs_adv) # (batch, num_classes)
                        
                        # Get predicted class indices
                        _, predicted_class_indices = torch.max(outputs_class_scores, 1)
                        
                        correct_predictions += (predicted_class_indices == true_class_indices).sum().item()
                        total_samples += true_class_indices.size(0)

                accuracy = 100. * correct_predictions / total_samples
                print(f"    Accuracy on {att_name} (gamma={gamma_val}): {accuracy:.2f}% ({correct_predictions}/{total_samples})")

    print("\nEvaluation finished.")