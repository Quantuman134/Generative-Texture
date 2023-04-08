import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device
import utils

'''ProgressivePoisitionEncoding from text2mesh, I have not understood it totally...
class ProgressiveEncoding(nn.Module):
    def __init__(self, mapping_size, T, d=3, progressive_enable=True):
        super().__init__()
        self._t = 0
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)])
        self.progressive_enable = progressive_enable
    
    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices)/self._tau).clamp(0, 1).repeat(2)
        if not self.progressive_enable:
            alpha = torch.ones_like(alpha)
        alpha = torch.cat([torch.ones(self.d), alpha], dim=0)
        self._t += 1
        return x * alpha
'''

################################
#the position encoding layer
class PositionEncoding(nn.Module):
    def __init__(self, input_dim=2, upper_freq_index=10) -> None:
        super().__init__()
        self.upper_freq_index = upper_freq_index
        self.freq_indices = torch.tensor([i for i in range(upper_freq_index)], device=device).repeat(input_dim)
        self.mapping_size = input_dim*2*upper_freq_index + input_dim

    def forward(self, x):
        x_input = x
        if x.dim() == 1:
            x_input = x.unsqueeze(0)
        x = x.repeat(1, self.upper_freq_index)
        x = torch.mul(x, pow(2, self.freq_indices)) * torch.pi
        return torch.cat((x_input, torch.sin(x), torch.cos(x)), dim=1).squeeze()
        
################################
# mlp representing a texture image
# output: color clamped within [-1, 1]    
class NeuralTextureField(nn.Module):
    def __init__(self, width, depth, input_dim=2, pixel_dim=3, pe_enable=True) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.pe_enable = pe_enable
        layers = []
        
        if pe_enable:
            pe = PositionEncoding(input_dim=input_dim)
            layers.append(pe)
            layers.append(nn.ReLU())
            layers.append(nn.Linear(pe.mapping_size, width))
        else:
            layers.append(nn.Linear(input_dim, width))  
        layers.append(nn.ReLU())
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, pixel_dim))
        self.base = nn.ModuleList(layers)

        print(self.base)
    
    def reset_weights(self):
        self.base[-1].weight.data.zero_()
        self.base[-1].bias.data.zero_()

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        colors = x

        #tanh clamp
        #colors = F.tanh(colors)
        return colors

def main():
    test_mlp = NeuralTextureField(width=256, depth=6)
    test_mlp.reset_weights()
    print(torch.pi)

if __name__ == "__main__":
    main()