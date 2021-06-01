"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.Layers import *

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()

    # q(y|x)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        GumbelSoftmax(128, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        Gaussian(128, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    return x

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)  
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat
  
  def forward(self, x, temperature=1.0, hard=0):
    #x = Flatten(x)

    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)
    
    # q(z|x,y)
    mu, var, z = self.qzxy(x, y)

    output = {'mean': mu, 'var': var, 'gaussian': z, 
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output


# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet, self).__init__()

    # p(z|y)
    self.y_mu = nn.Linear(y_dim, z_dim)
    self.y_var = nn.Linear(y_dim, z_dim)

    # p(x|z)
    self.generative_pxz = torch.nn.ModuleList([
        nn.Softmax(dim=1),
        nn.Linear(z_dim, x_dim, bias=False),
        nn.BatchNorm1d(x_dim, affine=True, eps=0.001, momentum=0.001),
        nn.Softmax(dim=1),
    ])

  # p(z|y)
  def pzy(self, y):
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y))
    return y_mu, y_var
  
  # p(x|z)
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    return z

  def forward(self, z, y):
    # p(z|y)
    y_mu, y_var = self.pzy(y)
    
    # p(x|z)
    x_rec = self.pxz(z)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output

# GMVAE Network
class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GMVAENet, self).__init__()

    self.inference = InferenceNet(x_dim, z_dim, y_dim)
    self.generative = GenerativeNet(x_dim, z_dim, y_dim)

    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
          init.constant_(m.bias, 0) 

  def forward(self, x, temperature=1.0, hard=0):
    x = x.view(x.size(0), -1)
    out_inf = self.inference(x, temperature, hard)
    z, y = out_inf['gaussian'], out_inf['categorical']
    out_gen = self.generative(z, y)
    
    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output







class GenerativeNet_style(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet_style, self).__init__()

    # p(z|y)
    self.y_mu = nn.Linear(y_dim, z_dim)
    self.y_var = nn.Linear(y_dim, z_dim)

    # p(x|z)

    self.g1 = nn.Softmax(dim=1)

    self.g2 = nn.Linear(z_dim, x_dim, bias=False)
    self.style_L = nn.Parameter(torch.ones(y_dim, z_dim))
    self.style_R = nn.Parameter(torch.ones(y_dim, x_dim))

    self.g3 = nn.BatchNorm1d(x_dim, affine=True, eps=0.001, momentum=0.001)
    self.g4 = nn.Softmax(dim=1)

  # p(z|y)
  def pzy(self, y):
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y))
    return y_mu, y_var

  # p(x|z)
  def pxz(self, z, y):
    # for layer in self.generative_pxz:
    #   z = layer(z)
    z = self.g1(z)

    tmp_L = torch.matmul(y.detach(), self.style_L)
    tmp_R = torch.matmul(y.detach(), self.style_R)
    gamma = torch.matmul(tmp_R.unsqueeze(2), tmp_L.unsqueeze(1))
    new_weight = self.g2.weight.unsqueeze(0)*gamma
    # z = torch.matmul(new_weight, z.unsqueeze(2)) + self.g2.bias.unsqueeze(0).unsqueeze(2)
    z = torch.matmul(new_weight, z.unsqueeze(2))
    z = z.squeeze()

    z = self.g3(z)
    z = self.g4(z)
    return z

  def forward(self, z, y):
    # p(z|y)
    y_mu, y_var = self.pzy(y)

    # p(x|z)
    x_rec = self.pxz(z, y)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output