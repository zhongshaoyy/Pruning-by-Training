import torch
import torch.nn as nn

from torch.autograd import Variable
torch.manual_seed(1)


# def Binarize(x, t=0.5):
#     #### ternary
#     # mask = (x > 0.5).float() - (x < -0.5).float()
#     # return mask

#     #### binary
#     return torch.where(torch.abs(x) < t, torch.full_like(x, 0), torch.full_like(x, 1)) 

def Binarize(x, alpha, t, is_training):

    if is_training:

        return alpha*torch.where(abs(x)<t, torch.full_like(x, 0),torch.full_like(x, 1)) + (1-alpha)*x 
    else:

        return torch.where(abs(x)<t, torch.full_like(x, 0),torch.full_like(x, 1))

# def Binarize(x, alpha, is_training):


#     if is_training:

#         return alpha*(torch.sign(x)+1) + (1-alpha)*x 
#     else:

#         return torch.sign(x)+1

class BinarizeAttention(nn.Module):
    def __init__(self, inplanes, alpha, t):
        super(BinarizeAttention, self).__init__()
        self.alpha= alpha
        self.t = t
        self.weight = nn.Parameter(torch.randn(inplanes, 1, 1), requires_grad=True)

    def forward(self, x):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = Binarize(self.weight.org, self.alpha, self.t, self.training)
        if x.device != torch.device('cpu'):
            self.weight.data = self.weight.data.cuda()
        return torch.mul(self.weight, x)

