


import torch
from torch import nn

class Conv1D_style(nn.Module):
    def __init__(self,  nx, nf, Num_cluster):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__()
        self.nf = nf
        w = torch.empty(nf, nx)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

        self.style_L = nn.Parameter(torch.ones(Num_cluster, nx))
        self.style_R = nn.Parameter(torch.ones(Num_cluster, nf))

    def forward(self, x, cluster):

        tmp_L = torch.matmul(cluster, self.style_L)
        tmp_R = torch.matmul(cluster, self.style_R)
        gamma = torch.matmul(tmp_L.unsqueeze(2), tmp_R.unsqueeze(1))

        new_weight = torch.transpose(self.weight.unsqueeze(0), 2, 1) * gamma
        x = torch.matmul(x, new_weight) + self.bias.unsqueeze(0).unsqueeze(1)

        return x

