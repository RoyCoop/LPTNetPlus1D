import numpy as np
import torch
import utils as utls
import torch.optim as optim


def train_model_1d(train_loader, valid_loader, model, criterion, optimizer, opt, device, l1_lambda=5e-10,
                   num_epochs=25):
    model.train()  # Set model to training mode
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.1)
    starting_epoch = 3
    K_array = utls.generate_K(num_epochs, starting_epoch)
    highest_certain_indices = []
    for epoch in range(num_epochs):
        if epoch == starting_epoch:
            model.uni_samp_map_flag = True
        print(f'Epoch {epoch + 1}/{num_epochs}')
        running_loss = 0.0
        batches = len(train_loader)
        all_samp_maps = []
        for i, (inputs, lengths) in enumerate(train_loader):
            print(f'Batch {i + 1}/{batches}')
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs, samp_map = model(inputs, return_samp_map=True, highest_certain_indices=highest_certain_indices)
            loss = criterion(outputs, inputs)

            # Compute L1 regularization for samp_map_generator parameters
            l1_penalty = utls.l1_regularization(samp_map, l1_lambda)

            print(f'loss is {loss}')
            print(f'l1 is {l1_penalty}')

            # Add L1 penalty to the loss
            total_loss = loss + l1_penalty

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * inputs.size(0)
            # Collect samp_map
            all_samp_maps.append(samp_map.detach().cpu().numpy())

        all_samp_maps = np.concatenate(all_samp_maps, axis=0)

        if model.uni_samp_map_flag:
            model.uni_samp_map, highest_certain_indices = utls.generate_universal_samp_map(all_samp_maps,
                                                                                           K_array[epoch])
            model.uni_samp_map = model.uni_samp_map.to(device)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.8f}')

        # Validate the model
        model.eval()
        val_loss = 0.0
        batches = len(valid_loader)
        with torch.no_grad():
            for i, (inputs, lengths) in enumerate(valid_loader):
                print(f'Batch {i + 1}/{batches}')
                inputs = inputs.to(device)
                outputs, samp_map = model(inputs, eval_flag=True, return_samp_map=True)
                l1_penalty = utls.l1_regularization(samp_map, l1_lambda)
                loss = criterion(outputs, inputs) + l1_penalty
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(valid_loader.dataset)
        print(f'Validation Loss: {val_loss:.8f}')
        print(scheduler.get_last_lr())
        scheduler.step(val_loss)

        model.train()  # Set model back to training mode after validation

    # utls.save_training_log(optimizer, model)


def evaluate_model_1d(test_loader, model, criterion, device, l1_lambda=5e-10):
    model.eval()
    test_loss = 0.0
    test_SAM = 0.0
    test_PSNR = 0.0
    test_QBE = 0.0

    with torch.no_grad():
        for inputs, lengths in test_loader:
            inputs = inputs.to(device)
            outputs, samp_map = model(inputs, eval_flag=True, return_samp_map=True)

            # Calculate L1 regularization penalty
            l1_penalty = utls.l1_regularization(samp_map, l1_lambda)

            # Calculate loss
            loss = criterion(outputs, inputs) + l1_penalty
            SAM = utls.spectral_angle_mapper(inputs, outputs)

            # Calculate PSNR using the separate function
            psnr = utls.calculate_psnr(outputs, inputs)

            # Calculate QBE
            qbe = utls.quantile_based_error(outputs, inputs)

            # Accumulate losses and PSNR
            test_loss += loss.item() * inputs.size(0)
            test_SAM += SAM * inputs.size(0)
            test_PSNR += psnr.item() * inputs.size(0)
            test_QBE += qbe.item() * inputs.size(0)

    # Average over all samples in the test dataset
    test_loss /= len(test_loader.dataset)
    test_SAM /= len(test_loader.dataset)
    test_PSNR /= len(test_loader.dataset)
    test_QBE /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.9f}')
    print(f'Test SAM Loss: {test_SAM:.9f}')
    print(f'Test PSNR: {test_PSNR:.9f} dB')
    print(f'Test QBE: {test_QBE:.9f} %')

    return test_PSNR, test_SAM, test_QBE, test_loss
