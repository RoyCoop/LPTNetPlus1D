import os
import numpy as np
import datetime
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import torch.nn.functional as F


def custom_collate_fn(batch):
    batch = [item for item in batch if item.shape[0] > 0]  # Remove empty items if any
    lengths = torch.tensor([item.shape[0] for item in batch], dtype=torch.long)
    padded_batch = pad_sequence(batch, batch_first=True)
    return padded_batch, lengths


def generate_universal_samp_map(all_samp_maps, K):
    # Reshape all_samp_maps to [total_samples, input_dim] if needed
    all_samp_maps = all_samp_maps.reshape(all_samp_maps.shape[0], -1)

    # Calculate the variance across the samples for each index (axis 0 is along samples)
    variances = np.var(all_samp_maps, axis=0)

    # Calculate the mean across the samples for each index
    mean_samp_map = np.mean(all_samp_maps, axis=0)

    # Get indices of the K smallest variances (most certain)
    highest_certain_indices = None
    if K != 0:
        sorted_indices = np.argsort(variances)
        highest_certain_indices = sorted_indices[:K]

    # Create a new universal sample map
    universal_samp_map = mean_samp_map

    # Convert to tensor and reshape to [1, input_dim]
    universal_samp_map = torch.tensor(universal_samp_map).reshape(1, -1)

    return universal_samp_map, highest_certain_indices


def generate_K(num_epochs, start):
    # Initialize variables
    dim = 6950
    K_initial = int(np.round(dim * 0.1))  # Initial K value as an integer

    # Define a non-linear progression (e.g., exponential)
    x = np.linspace(0, 1, num_epochs - start)  # Linearly spaced values between 0 and 1
    progression = np.power(x, 2)  # Use a quadratic curve for slower start and faster finish

    # Scale the progression to fit the range between K_initial and dim
    K_array = K_initial + progression * (dim - K_initial)

    # Round and convert to integers
    K_array = np.round(K_array).astype(int)

    # Ensure the last value is exactly equal to dim
    K_array[-1] = dim

    # Creating a 3x1 array of zeros
    zeros_array = np.zeros([start, ])

    # Concatenate along the first axis (vertically)
    K_array = np.concatenate((zeros_array, K_array)).astype(int)

    return K_array


def l1_regularization(parameters, l1_lambda):
    l1_norm = sum(p.abs().sum() for p in parameters)
    return l1_lambda * l1_norm


def spectral_angle_mapper(pred, target):
    # Ensure the spectra are of the same shape
    assert pred.shape == target.shape, "Predictions and targets must have the same shape"

    # Normalize the predictions and targets
    pred_norm = torch.norm(pred, dim=1, keepdim=True)  # Norm along the feature axis
    target_norm = torch.norm(target, dim=1, keepdim=True)

    epsilon = 1e-8  # Small constant to prevent division by zero
    pred_normalized = pred / (pred_norm + epsilon)
    target_normalized = target / (target_norm + epsilon)

    # Compute the cosine similarity
    cos_theta = torch.sum(pred_normalized * target_normalized, dim=1)  # Dot product along the feature dimension

    # Clamp the cosine values to avoid numerical instability
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute the spectral angle in radians
    sam_loss = torch.acos(cos_theta)

    # Return the mean SAM loss across the batch
    return torch.mean(sam_loss)


def calculate_psnr(pred, target, max_val=1.0):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between the prediction and the target.

    Parameters:
    - pred: The predicted tensor (output of the model)
    - target: The ground truth tensor (input data)
    - max_val: The maximum possible value in the data (default is 1.0 for normalized data)

    Returns:
    - psnr: The PSNR value in decibels (dB)
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr



def quantile_based_error(pred, target, quantile=0.95):
    """
    Calculate the Quantile-based Error (QBE) between the prediction and the target.

    Parameters:
    - pred: The predicted tensor (output of the model)
    - target: The ground truth tensor (input data)
    - quantile: The quantile to use for error evaluation (default is 0.95 for 95th percentile)

    Returns:
    - qbe: The Quantile-based Error value
    """
    # Ensure the predictions and targets have the same shape
    assert pred.shape == target.shape, "Predictions and targets must have the same shape"

    # Calculate the absolute errors
    errors = torch.abs(pred - target)

    # Flatten the errors to get a 1D tensor
    errors_flat = errors.view(-1)

    # Compute the quantile value
    qbe = torch.quantile(errors_flat, quantile)

    return qbe


def plot_intermediate_results(intermediate_results, inputs, output_path, model_M):
    rand_entry = np.random.randint(len(intermediate_results))
    batch_size = intermediate_results[0][0].shape[0]
    rand_sample = np.random.randint(batch_size)
    batch_start_index = rand_entry - (rand_entry % batch_size)
    x_model_path = []
    x_original = inputs[rand_sample, 0]
    for x, x_unet, x_after in intermediate_results[batch_start_index:batch_start_index + 3]:
        x_np = x.squeeze()[rand_sample, :]
        x_np = x_np.detach().cpu().numpy().T
        x_unet_np = x_unet.squeeze()[rand_sample, :]
        x_unet_np = x_unet_np.detach().cpu().numpy().T
        x_after_np = x_after.squeeze()[rand_sample, :]
        x_after_np = x_after_np.detach().cpu().numpy().T
        x_model_path.extend([x_np, x_unet_np, x_after_np, x_original])

    # Create an index array for the X-axis
    indices = np.arange(1, 6951)  # from 1 to 6950

    # Create a figure with 3 subplots (one for each block)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Define styles and colors for each block
    styles = ['-', '--', ':']
    colors = ['blue', 'green', 'red', 'black']

    # Plot each block's results in a separate subplot
    for i in range(0, 9, 3):
        block_num = i // 3 + 1  # Block number: 1, 2, 3
        ax = axes[block_num - 1]  # Select the correct subplot (0-based index)

        # Plot the data for each block in its subplot
        ax.plot(indices, x_model_path[i], label=f'x_block_{block_num}', linestyle=styles[0], color=colors[0], alpha=0.6)
        ax.plot(indices, x_model_path[i + 1], label=f'x_unet_block_{block_num}', linestyle=styles[0], color=colors[1],
                alpha=0.6)
        ax.plot(indices, x_model_path[i + 2], label=f'x_after_block_{block_num}', linestyle=styles[0], color=colors[2],
                alpha=0.6)
        ax.plot(indices, x_model_path[i + 3], label=f'x_original', linestyle=styles[0], color=colors[3],
                alpha=0.6)

        # Customize each subplot
        ax.set_title(f'Block {block_num}')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend(loc='best')

    # Common X label
    axes[-1].set_xlabel('Index')

    # Adjust layout
    plt.tight_layout()
    plot_filename = os.path.join(output_path, f'Intermediate_Results_Subplots_M{model_M}.png')
    fig.savefig(plot_filename)
    plt.close(fig)


def save_training_log(optimizer, model, base_path='/content/drive/MyDrive/EE_Project/output_plots'):
    # Define base output directory
    base_output_dir = base_path

    # Get the current date to use in the folder name
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    # Define the folder name
    folder_name = f"{date_str} model training_M{model.M}"

    # Create the full path
    output_dir = os.path.join(base_output_dir, folder_name)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a directory for weight histograms
    weight_histogram_dir = os.path.join(output_dir, "weight_histograms")
    os.makedirs(weight_histogram_dir, exist_ok=True)

    # Plot and save weight histograms
    plot_weights_histogram(model, weight_histogram_dir)

    # Save optimizer parameters
    opt_params_file = os.path.join(output_dir, "opt_parameters.txt")
    save_opt_params(optimizer, opt_params_file)

    # Save gradient information
    grad_file = os.path.join(output_dir, "gradient_prints.txt")
    save_gradients(model, grad_file)

    # Inform that all files have been saved
    print("All outputs have been saved successfully.")


# Function to save optimizer parameters
def save_opt_params(optimizer, filepath):
    with open(filepath, 'w') as f:
        for param_group in optimizer.param_groups:
            f.write(f"Learning Rate: {param_group['lr']}\n")
            f.write(f"Weight Decay: {param_group.get('weight_decay', 0)}\n")
            # Add other hyperparameters if needed
            f.write("\n")  # Separate different param groups with a newline


# Function to save gradients
def save_gradients(model, filepath):
    with open(filepath, 'w') as f:
        for name, param in model.named_parameters():
            if param.grad is not None:
                f.write(f"Parameter: {name}, Gradient Norm: {param.grad.data.norm().item()}\n")
            else:
                f.write(f"Parameter: {name}, Gradient: None\n")


# Function to plot and save weight histograms
def plot_weights_histogram(model, filepath):
    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.figure(figsize=(6, 4))
            plt.hist(param.data.cpu().numpy().flatten(), bins=50, color='blue', alpha=0.7)
            plt.title(f"Histogram of {name}")
            plt.xlabel("Weight values")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(filepath, f"{name}_weights_histogram.png"))
            plt.close()


def save_model(model, model_name, model_M, save_model_path='/content/drive/MyDrive/EE_Project/saved_models_1d',
               epoch=None):
    # Get the current date
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    # Define the path where you want to save the model
    save_path = os.path.join(save_model_path, date_str)

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the model
    if epoch is not None:
        model_filename = f'{model_name}_epoch{epoch}_M{model_M}.pth'
    else:
        model_filename = f'{model_name}_M{model_M}_final.pth'
    model_save_path = os.path.join(save_path, model_filename)
    torch.save(model, model_save_path)

    print(f"Model saved at: {model_save_path}")


def load_model(model_path, device):
    # Load the entire model object onto the specified device
    model = torch.load(model_path, map_location=device)

    # Ensure the model is moved to the specified device
    model.to(device)

    # Set the model to evaluation mode if needed
    model.eval()

    return model
