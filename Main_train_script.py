import utils as utls
import optml_samp_utils
import LSCSnet_model as models
import LSCSnet_data_loaders as data_loaders
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device('cuda')
    device_flag = 'cuda'
else:
    device = torch.device('cpu')
    device_flag = 'cpu'
print(f'Device used: {device_flag}')

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, help='Path to the dataset',
                    default='C:\\Users\\royco\\PycharmProjects\\CSproject\\Learned Sensing coefficients CS '
                            'net\\small_split_data')
parser.add_argument('--save_model_path', type=str, help='trained model path',
                    default='C:\\Users\\royco\\PycharmProjects\\CSproject\\Learned Sensing coefficients CS '
                            'net\\saved_models_1d')
parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=5)
parser.add_argument('--M', type=int, help='Maximum number of Fourier coefficients', default=500)
parser.add_argument('--l1_lambda', type=float, help='L1 loss lambda', default=7e-11)
opt = parser.parse_args()
print(opt)


def main_1d():
    train_folder = os.path.join(opt.data_folder, 'train')
    valid_folder = os.path.join(opt.data_folder, 'valid')
    test_folder = os.path.join(opt.data_folder, 'test')

    train_dataset = data_loaders.SpectrogramDataset(train_folder)
    valid_dataset = data_loaders.SpectrogramDataset(valid_folder)
    test_dataset = data_loaders.SpectrogramDataset(test_folder)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                              collate_fn=utls.custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True,
                              collate_fn=utls.custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True,
                             collate_fn=utls.custom_collate_fn)

    model = models.LSCSNet_base_1d(M=opt.M).to(device)
    models.initialize_weights(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': model.MSCSNet_model.parameters(), 'lr': opt.learning_rate},
        {'params': model.samp_map_generator.parameters(), 'lr': opt.learning_rate}
    ])

    # Train the model
    optml_samp_utils.train_model_1d(train_loader, valid_loader, model, criterion, optimizer, opt, device,
                                    num_epochs=opt.epochs, l1_lambda=opt.l1_lambda)

    # Save the model
    utls.save_model(model, "lptnet1D", model.M, opt.save_model_path)

    # Evaluate on test set
    optml_samp_utils.evaluate_model_1d(test_loader, model, criterion, device, l1_lambda=opt.l1_lambda)


if __name__ == "__main__":
    main_1d()
