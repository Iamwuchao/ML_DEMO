import numpy as np
import matplotlib.pyplot as plt
from IPython import display

plt.style.use('seaborn-white')
data = open('input.txt', 'r').read()

chars = list(set(data))
data_size,X_size = len(data),len(chars)
print("data has %d characters,%d unique" %(data_size,X_size))
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

H_size = 100
T_steps = 25
learning_rate = 25
weight_sd = 0.1
z_size = H_size + X_size

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):


def dsigmoid(y):
    return y*(1-y)

def dtanh(y):
    return 1-y*y

class Param:
    def __init__(self,name,value):
        self.name = name
        self.v = value
        self.d = np.zeros_like(value)
        self.m = np.zeros_like(value)

class Parameters:
    def __init__(self):
        self.W_f = Param('W_f',np.random.randn(H_size)*weight_sd+0.5)
        self.b_f = Param('b_f',np.zeros((H_size,1)))
        self.W_i = Param('W_i',np.random.randn(H_size,z_size)*weight_sd+0.5)
        self.b_i = Param('b_i',np.zeros(H_size,1))
        self.W_C = Param('W_C',np.random.randn(H_size,z_size)*weight_sd)
        self.b_C = Param('b_C',np.zeros((H_size,1)))
        self.W_o = Param('W_o',np.random.randn(H_size,z_size)*weight_sd)
        self.b_o = Param('b_o',np.zeros(H_size,1))
        self.W_v = Param('W_v',np.random.randn(X_size,H_size)*weight_sd)
        self.b_v = Param('b_v',np.zeros((X_size,1)))

    def all(self):
        return [self.W_f,self.W_i,self.W_c,self.W_o,self.W_v,self.b_f,self.b_i,
                self.b_C,self.b_o,self.b_v]

parameters = Parameters()

def forward(x,h_prev,C_prev,p = parameters):
    assert x.shape == (X_size,1)
    assert h_prev.shape == (H_size,1)
    assert  C_prev.shape == (H_size,1)
    z = np.row_stack((h_prev,x))
    f = sigmoid(np.dot(p.W_f.v,z)+p.b_f.v)
    i = sigmoid(np.dot(p.W_i.v,z)+p.b_i.v)
    C_bar = tanh(np.dot(p.W_C.v,z)+p.b_C.v)
    C = f*C_prev + i* C_bar
    o = sigmoid(np.dot(p.W_o.v,z)+p.b_o.v)
    h = o*tanh(C)
    v = np.dot(p.W_v.v,h) + p.b_v.v
    y = np.exp(v)/np.sum(np.exp(v)) #softmax
    return z,f,i,C_bar,C,o,h,v,y

def backward(target,dh_next,dC_next,C_prev,z,f,i,C_bar,C,o,h,v,y,p=parameters):
    assert z.shape ==(X_size+H_size,1)
    assert v.shape ==(X_size,1)
    assert y.shape == (X_size,1)

    for param in [dh_next,dC_next,C_prev,f,i,C_bar,C,o,h]:
        assert param.shape ==(H_size,1)
    dv = np.copy(y)
    dv[target]-=1
    p.W_v.d += np.dot(dv,h.T)
    p.b_v.d +=dv

    dh = np.dot(p.W_v.v.T,dv)
    dh += dh_next
    do = dh*tanh(C)
    do = dsigmoid(o)*do
    p.W_o.d += np.dot(do,z.T)
    p.b_o.d += do

    dC = np.copy(dC_next)
    dC += dh*o*dtanh(tanh(C))
    dC_bar = dC*i
    dC_bar = dtanh(C_bar)

















