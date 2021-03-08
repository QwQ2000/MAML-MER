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
        self.vars = nn.ParameterList()
        self.dict = dict()

    def zero_grad(self,vars = None):
        with torch.no_grad():
            v = self.vars if vars is None else vars
            for p in v:
                if p.grad is not None:
                    p.grad.zero_()
    
    def parameters(self):
        return self.vars

class STSTNet(FunctionalNet):
    def __init__(self):
        super(STSTNet,self).__init__()
        d = dict()
        d['conv1_w'],d['conv1_b'] = getParams(layer_type = 'conv2d',out_c = 3,in_c = 3,ker_sz = 3)
        d['conv2_w'],d['conv2_b'] = getParams(layer_type = 'conv2d',out_c = 5,in_c = 3,ker_sz = 3)
        d['conv3_w'],d['conv3_b'] = getParams(layer_type = 'conv2d',out_c = 8,in_c = 3,ker_sz = 3)
        d['fc1_w'],d['fc1_b'] = getParams(layer_type = 'linear',out_c = 400,in_c = 400)
        d['fc2_w'],d['fc2_b'] = getParams(layer_type = 'linear',out_c = 5,in_c = 400)
        for k,v in d.items():
            self.vars.append(v)
            self.dict[k] = len(self.vars) - 1

    def forward(self,x,vars = None,train = True):
        if vars is None:
            vars = self.vars
        gp = lambda name : vars[self.dict[name]]
        x1 = F.conv2d(x,gp('conv1_w'),gp('conv1_b'),stride = 1,padding = 1)
        x1 = F.max_pool2d(x1,kernel_size = 3,stride = 3,padding = 1)
        x2 = F.conv2d(x,gp('conv2_w'),gp('conv2_b'),stride = 1,padding = 1)
        x2 = F.max_pool2d(x2,kernel_size = 3,stride = 3,padding = 1)
        x3 = F.conv2d(x,gp('conv3_w'),gp('conv3_b'),stride = 1,padding = 1)
        x3 = F.max_pool2d(x3,kernel_size = 3,stride = 3,padding = 1)
        x = torch.cat([x1,x2,x3],dim = 1)
        x = F.max_pool2d(x,kernel_size = 2,stride = 2)
        x = x.view(-1,400)
        x = F.linear(x,gp('fc1_w'),gp('fc1_b'))
        x = F.relu(x)
        x = F.linear(x,gp('fc2_w'),gp('fc2_b'))
        return x