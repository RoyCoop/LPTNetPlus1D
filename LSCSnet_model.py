import torch
import torch.nn as nn
import UNet1D
import torch.fft


class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        # We want the attention layer to produce exactly M scores.
        self.attention_layer = nn.Linear(input_dim, attention_dim)  # attention_dim should be M
        self.context_layer = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        # Linear transformation followed by a non-linearity
        attention_scores = self.attention_layer(x)
        attention_scores = torch.tanh(attention_scores)

        # Calculate attention weights and normalize
        attention_weights = self.context_layer(attention_scores)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        return attention_weights


class SampMapGenerator(nn.Module):
    def __init__(self, input_dim=6950, output_dim=6950, M=500):
        super(SampMapGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, M)
        self.bn2 = nn.BatchNorm1d(M)
        self.attention = Attention(M, M)  # Attention layer produces M scores
        self.fc3 = nn.Linear(M, output_dim)
        self.relu = nn.ReLU()  # ReLU activation

    def forward(self, x):
        # Transform input to frequency domain
        x_freq = torch.fft.fft(x, dim=-1).abs()  # Use absolute value of FFT to get magnitude
        x = x_freq.view(x.size(0), -1)  # Flatten to (batch_size, input_dim)

        # Forward through fully connected layers
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights  # Element-wise multiplication

        x = self.fc3(x)
        x = self.relu(x)

        # Normalize to [0, 1] range
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
        samp_map = (x - x_min) / (x_max - x_min + 1e-8)

        samp_map = samp_map.view(-1, 1, 6950)
        return samp_map


class LSCSNet1D(nn.Module):
    def __init__(self, n_in_c=1):
        super(LSCSNet1D, self).__init__()
        self.unet = UNet1D.UNet1D(n_in_c, n_in_c)

    def forward(self, x, samp_map, y):
        # Apply U-Net
        x_unet = self.unet(x)
        # Perform FFT
        y1 = torch.fft.fft(x_unet)
        # Calculate w
        y2 = torch.mul(y1, (1 - samp_map)) + y
        return y2, x_unet


class LSCSNet_base_1d(nn.Module):
    def __init__(self, n_in_c=1, iters_no=3, M=500, uni_samp_map_flag=False, uni_samp_map=None):
        super().__init__()
        self.MSCSNet_model = nn.ModuleList([
            LSCSNet1D(n_in_c=n_in_c) for _ in range(iters_no)
        ])
        self.y_dict = {}
        self.samp_map_generator = SampMapGenerator(input_dim=6950, output_dim=6950, M=M)
        self.M = M
        self.uni_samp_map = uni_samp_map
        self.uni_samp_map_flag = uni_samp_map_flag

    def forward(self, x, return_samp_map=False, highest_certain_indices=None, eval_flag=False, intermediate_flag=False):
        if self.uni_samp_map_flag and self.uni_samp_map is not None:
            if not eval_flag:
                with torch.no_grad():
                    dynamic_samp_map = self.samp_map_generator(x)
                    dynamic_samp_map[:, :, highest_certain_indices] = self.uni_samp_map[:,
                                                                      highest_certain_indices].unsqueeze(0)
                    samp_map = dynamic_samp_map
            else:
                samp_map = self.uni_samp_map
        else:
            samp_map = self.samp_map_generator(x)

        binary_samp_map = self.get_binary_samp_map(samp_map)
        y_initial = torch.fft.fft(x, dim=-1) * samp_map
        y = y_initial
        intermediate_results = []

        for i, model_net in enumerate(self.MSCSNet_model):
            x = torch.fft.ifft(y, dim=-1).real
            self.y_dict[i] = y
            y, x_unet = model_net(x, samp_map, y_initial)
            intermediate_results.append((x, x_unet, torch.fft.ifft(y, dim=-1).real))

        y = y * binary_samp_map
        x = torch.fft.ifft(y, dim=-1).real

        if return_samp_map:
            if not eval_flag:
                return x, samp_map
            else:
                if intermediate_flag:
                    return x, binary_samp_map, intermediate_results
                else:
                    return x, binary_samp_map
        else:
            return x

    def get_binary_samp_map(self, samp_map):
        # Ensure samp_map has exactly M ones
        binary_samp_map = torch.zeros_like(samp_map)
        topk = torch.topk(samp_map, self.M, dim=-1)
        binary_samp_map.scatter_(-1, topk.indices, 1)
        return binary_samp_map


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization for ReLU
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
