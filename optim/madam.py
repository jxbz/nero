import math
import torch
from torch.optim.optimizer import Optimizer, required

import numpy as np

def ConstantSum(x,a=0.5,b=100,sum_axis=1,norm_axis=0, p=0.5):
  if len(x.shape) == 4: 
    x_pos = torch.clamp(x,min=0)
    x_neg = torch.clamp(x,max=0)
    x_pos_sum = torch.sum(x_pos,(sum_axis,2,3))
    x_neg_sum = -torch.sum(x_neg,(sum_axis,2,3))
    x = a * ((x.shape[norm_axis] *x.shape[2] *x.shape[3]) ** p) * (x_pos / x_pos_sum.view(x_pos.shape[0], 1, 1, 1) 
        + x_neg / x_neg_sum.view(x_pos.shape[0], 1, 1, 1))
  elif len(x.shape) == 2: 
    x_pos = torch.clamp(x,min=0)
    x_neg = torch.clamp(x,max=0)
    x_pos_sum = torch.sum(x_pos,sum_axis)
    x_neg_sum = -torch.sum(x_neg,sum_axis)
    x = a * (x.shape[norm_axis] ** p) * (x_pos / x_pos_sum.view(x_pos.shape[0], 1) + x_neg / x_neg_sum.view(x_pos.shape[0], 1))
  elif x.ndim == 1:
    x = torch.clamp(x,min=-b,max=b)
  return x

class Madam(Optimizer):

    def __init__(self, params, lr=0.01, p_scale=3.0, g_bound=10.0, a=0.5, p=0.5,sum_flag=False,interval=2):
        self.interval=interval
        self.sum_flag = sum_flag
        self.a = a
        self.p = p
        self.p_scale = p_scale
        self.g_bound = g_bound
        defaults = dict(lr=lr)
        super(Madam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['max'] = self.p_scale*(p*p).mean().sqrt().item()
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                bias_correction = 1 - 0.999 ** state['step']
                state['exp_avg_sq'] = 0.999 * state['exp_avg_sq'] + 0.001 * p.grad.data**2
                
                g_normed = p.grad.data / (state['exp_avg_sq']/bias_correction).sqrt()
                g_normed[torch.isnan(g_normed)] = 0
                g_normed.clamp_(-self.g_bound, self.g_bound)
                
                p.data *= torch.exp( -group['lr']*g_normed*torch.sign(p.data) )
                p.data.clamp_(-state['max'], state['max'])
                if self.sum_flag:
                    p.data = ConstantSum(p.data,a=self.a,p=self.p)
        return loss
