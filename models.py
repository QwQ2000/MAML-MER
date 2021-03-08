import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def getParams(**kargs):
    if kargs['layer_type'] == 'conv2d':
        w = nn.Parameter(torch.ones(kargs['out_c'],kargs['in_c'],kargs['ker_sz'],kargs['ker_sz']))
        torch.nn.init.kaiming_normal_(w)
        b = nn.Parameter(torch.zeros(kargs['out_c']))
        return w,b
    elif kargs['layer_type'] == 'linear':
        w = nn.Parameter(torch.ones(kargs['out_c'],kargs['in_c']))
        torch.nn.init.kaiming_normal_(w)
        b = nn.Parameter(torch.zeros(kargs['out_c']))
        return w,b

class FunctionalNet(nn.Module):
    def __init__(self):
        super(FunctionalNet,self).__init__()
        self.vars = nn.ParameterDict()
    
    def zero_grad(self,vars = None):
        with torch.no_grad():
            v = self.vars if vars is None else vars
            for p in v.values():
                if p.grad is not None:
                    p.grad.zero_()
    
    def parameters(self):
        return self.vars

class STSTNet(FunctionalNet):
    def __init__(self):
        super(STSTNet,self).__init__()
        self.vars['conv1_w'],self.vars['conv1_b'] = getParams(layer_type = 'conv2d',out_c = 3,in_c = 3,ker_sz = 3)
        self.vars['conv2_w'],self.vars['conv2_b'] = getParams(layer_type = 'conv2d',out_c = 5,in_c = 3,ker_sz = 3)
        self.vars['conv3_w'],self.vars['conv3_b'] = getParams(layer_type = 'conv2d',out_c = 8,in_c = 3,ker_sz = 3)
        self.vars['fc1_w'],self.vars['fc1_b'] = getParams(layer_type = 'linear',out_c = 400,in_c = 400)
        self.vars['fc2_w'],self.vars['fc2_b'] = getParams(layer_type = 'linear',out_c = 3,in_c = 400)
    
    def forward(self,x,vars = None,train = True):
        if vars is None:
            vars = self.vars
        x1 = F.conv2d(x,vars['conv1_w'],vars['conv1_b'],stride = 1,padding = 1)
        x2 = F.conv2d(x,vars['conv2_w'],vars['conv2_b'],stride = 1,padding = 1)
        x3 = F.conv2d(x,vars['conv3_w'],vars['conv3_b'],stride = 1,padding = 1)
        x = torch.cat([x1,x2,x3],dim = 1)
        x = F.max_pool2d(kernel_size = 3,stride = 3,padding = 1)
        x = x.view(-1,400)
        x = F.linear(x,self.vars['fc1_w'],self.vars['fc1_b'])
        x = F.relu(x)
        x = F.linear(x,self.vars['fc2_w'],self.vars['fc2_b'])
        return x