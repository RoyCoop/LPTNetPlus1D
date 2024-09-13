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
import seaborn as sns


def visualize_results(test_loader, model, device, output_path, num_samples=5):
    model.eval()
    inputs, lengths = next(iter(test_loader))
    inputs = inputs.to(device)
    # bin = model.get_binary_samp_map(model.uni_samp_map)
    # inputs_F = torch.fft.fft(inputs, dim=-1) * bin
    # inputs_masked = torch.fft.ifft(inputs_F, dim=-1)
    outputs, binary_samp_map, intermediate_results = model(inputs, return_samp_map=True, eval_flag=True,
                                                           intermediate_flag=True)
    binary_samp_map = torch.fft.fftshift(binary_samp_map)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 2))
    for i in range(num_samples):
        axes[i, 0].plot(inputs[i, 0].cpu().numpy(), label='Original')
        axes[i, 0].set_title('Original Signal')

        axes[i, 1].plot(outputs[i, 0].cpu().detach().numpy(), label='Reconstructed')
        axes[i, 1].set_title('Reconstructed Signal')

        # Add the heatmap for the binary samp map
        sns.heatmap(binary_samp_map.cpu().reshape(1, -1), cmap="Greys", cbar=False, ax=axes[i, 2], xticklabels=1000,
                    yticklabels=False)

        axes[i, 2].set_title('Binary Samp Map')
        # Set axis labels
        axes[i, 2].set_xlabel('Frequency index')
        axes[i, 2].set_ylabel('Binary Value')

    plt.tight_layout()

    # Save the plot image
    plot_filename = os.path.join(output_path, f'output_M_{model.M}.png')
    fig.savefig(plot_filename)
    plt.close(fig)

    utls.plot_intermediate_results(intermediate_results, inputs, output_path, model.M)


def main(opt_model_path, opt_data_path, opt_output_path):
    # Define device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device used: {device}')

    model = utls.load_model(opt_model_path, device)

    # Load the test dataset
    test_folder = opt_data_path
    test_dataset = data_loaders.SpectrogramDataset(test_folder)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=utls.custom_collate_fn)

    # Visualize the results
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(opt_output_path, date_str)
    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists
    visualize_results(test_loader, model, device, output_path=output_path, num_samples=3)

    if False:
        # Define loss criterion
        criterion = nn.MSELoss()

        # Evaluate the model
        test_PSNR, test_SAM, test_QBE, test_loss = optml.evaluate_model_1d(test_loader, model, criterion, device)

        # Write the results to a text file
        results_file_path = os.path.join(output_path, f"evaluation_metrics_M{model.M}.txt")
        with open(results_file_path, "w") as f:
            f.write(f"Test PSNR: {test_PSNR}\n")
            f.write(f"Test SAM: {test_SAM}\n")
            f.write(f"Test QBE: {test_QBE}\n")
            f.write(f"Test Loss: {test_loss}\n")

        print(f"Evaluation metrics saved to {results_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model_path', type=str, help='Path to model',
                        default='C:\\Users\\royco\\PycharmProjects\\CSproject\\Learned Sensing coefficients CS net\\saved_models_1d\\2024-09-04\\lptnet1D_M800_final.pth')
    parser.add_argument('--test_folder', type=str, help='Path to test folder',
                        default='C:\\Users\\royco\\PycharmProjects\\CSproject\\Learned Sensing coefficients CS net\\split_data_absorption\\test')
    parser.add_argument('--output_path', type=str, help='Path to save outputs',
                        default='C:\\Users\\royco\\PycharmProjects\\CSproject\\Learned Sensing coefficients CS net\\output_plots')
    opt = parser.parse_args()
    main(opt.Model_path, opt.test_folder, opt.output_path)

# Colab run: !python /content/drive/MyDrive/EE_Project/Main_evaluate_script.py --Model_path /content/drive/MyDrive/EE_Project/output_plots --test_folder /content/drive/MyDrive/EE_Project/split_data/test --output_path /content/drive/MyDrive/EE_Project/output_plots
