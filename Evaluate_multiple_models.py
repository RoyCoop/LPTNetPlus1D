import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import LSCSnet_data_loaders as data_loaders
import utils as utls
import optml_samp_utils as optml
import torch.nn as nn
import datetime
import argparse


def evaluate_multiple_models(test_loader, model_paths, device):
    # Store results for each metric
    psnr_values = []
    sam_values = []
    l1_l2_values = []
    qbe_values = []
    M_values = []

    criterion = nn.MSELoss()

    for model_path in model_paths:
        # Extract M from model path
        M = int(model_path.split('_M')[1].split('_')[0])

        # Load model
        model = utls.load_model(model_path, device)

        # Evaluate model
        psnr, sam_loss, qbe, l1_l2_loss = optml.evaluate_model_1d(test_loader, model, criterion, device)

        # Store results
        psnr_values.append(psnr)
        sam_values.append(sam_loss.cpu())
        l1_l2_values.append(l1_l2_loss)
        qbe_values.append(qbe)
        M_values.append(M)

    return M_values, psnr_values, sam_values, qbe_values, l1_l2_values


def plot_metrics(M_values, psnr_values, sam_values, qbe_values, l1_l2_values, output_path):
    plt.figure(figsize=(14, 8))

    plt.subplot(4, 1, 1)
    plt.plot(M_values, psnr_values, marker='o')
    plt.title('PSNR vs M')
    plt.xlabel('M')
    plt.ylabel('PSNR (dB)')

    plt.subplot(4, 1, 2)
    plt.plot(M_values, sam_values, marker='o', color='orange')
    plt.title('SAM vs M')
    plt.xlabel('M')
    plt.ylabel('SAM')

    plt.subplot(4, 1, 3)
    plt.plot(M_values, qbe_values, marker='o', color='yellow')
    plt.title('QBE vs M')
    plt.xlabel('M')
    plt.ylabel('QBE')

    plt.subplot(4, 1, 4)
    plt.plot(M_values, l1_l2_values, marker='o', color='green')
    plt.title('L1 + L2 Loss vs M')
    plt.xlabel('M')
    plt.ylabel('L1 + L2 Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'model_comparison.png'))
    plt.show()


def main(opt_data_path, opt_output_path):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device used: {device}')

    # Define model paths
    model_paths = [f'/content/drive/MyDrive/EE_Project/saved_models_1d/2024-09-03/lptnet1D_M{M}_final.pth' for M in
                   [200,300,500]]
    model_paths2 = [f'/content/drive/MyDrive/EE_Project/saved_models_1d/2024-09-04/lptnet1D_M{M}_final.pth' for M in
                   [400,600,700,800,900,1000]]
    model_paths = model_paths+model_paths2

    # Load the test dataset
    test_folder = opt_data_path
    test_dataset = data_loaders.SpectrogramDataset(test_folder)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=utls.custom_collate_fn)

    # Evaluate all models and gather results
    M_values, psnr_values, sam_values, qbe_values, l1_l2_values = evaluate_multiple_models(test_loader, model_paths, device)

    # Plot the results
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(opt_output_path, date_str)
    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists
    plot_metrics(M_values, psnr_values, sam_values, qbe_values, l1_l2_values, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', type=str, help='Path to test folder',
                        default='/content/drive/MyDrive/EE_Project/split_data_absorption/test')
    parser.add_argument('--output_path', type=str, help='Path to save outputs',
                        default='/content/drive/MyDrive/EE_Project/output_plots')
    opt = parser.parse_args()
    main(opt.test_folder, opt.output_path)
