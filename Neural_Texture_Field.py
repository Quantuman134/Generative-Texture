import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0")
print(torch.cuda.get_device_name(0))
torch.cuda.set_device(device)
print(torch.cuda.get_device_name(0))

class ProgressiveEncoding(nn.Module):
    def __init__(self, mapping_size, T, d=3, progressive_enable=True):
        super().__init__()
        self._t = 0
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)
        self.progressive_enable = progressive_enable
    
    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices)/self._tau).clamp(0, 1).repeat(2)
        if not self.progressive_enable:
            alpha = torch.ones_like(alpha, device=device)
        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha
    
class NeuralTextureField(nn.Module):
    def __init__(self, width, depth, input_dim=2) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        layers = []
        
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        self.base = nn.ModuleList(layers)

        print(self.base)
    
    ''' bug exist, wait for fix
    def reset_weights(self):
        self.base[-1].weight.data.zero_()
        self.base[-1].bias.data.zero_()
    '''
    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        colors = x
        
        #tanh clamp
        colors = F.tanh(colors) / 2 + 0.5
        return colors

def main():
    test_mlp = NeuralTextureField(256, 6)
    test_mlp.reset_weights()
    #print(test_mlp.base[0].weight)

if __name__ == "__main__":
    main()