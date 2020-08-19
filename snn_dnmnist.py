import argparse, time

import torch
import numpy as np
from torchneuromorphic.doublenmnist.doublenmnist_dataloaders import *

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")
dtype = torch.float32
ms = 1e-3

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=72, help='Training Epochs')
parser.add_argument("--epochs", type=int, default=500, help='Training Epochs')
parser.add_argument("--burnin", type=int, default=20, help='Burnin Phase')
parser.add_argument("--lr", type=float, default=1.0e-7, help='Learning Rate')
parser.add_argument("--delta-t", type=int, default=1, help='Number of classes')

parser.add_argument("--nclasses", type=int, default=5, help='Number of classes')
parser.add_argument("--samples-per-class", type=int, default=100, help='Number of classes')


args = parser.parse_args()

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''
    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >= 0).float()

    def backward(aux, grad_output):
        #grad_input = grad_output.clone()        
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input


class LIF_FC_Layer(torch.nn.Module):
    def __init__(self, input_neurons, output_neurons, tau_syn, tau_mem, tau_ref, delta_t, bias, device, dtype):
        super(LIF_FC_Layer, self).__init__()   
        self.device = device
        self.dtype = dtype
        self.inp_neurons = input_neurons      
        self.out_neurons = output_neurons
        self.init =  np.sqrt(6 / self.inp_neurons) 
                
        self.weights = torch.nn.Parameter(torch.empty((self.inp_neurons, self.out_neurons),  device=device, dtype= dtype, requires_grad=True))
        torch.nn.init.uniform_(self.weights, a = -self.init, b = self.init)
        if bias:
            self.bias    = torch.nn.Parameter(torch.empty((self.out_neurons), device=device, dtype=dtype, requires_grad=True))
            torch.nn.init.uniform_(self.bias, a = -0, b = 0)
        else:
            self.bias = None
          
        # taus and betas
        self.beta = torch.tensor([1 - delta_t / tau_syn], dtype = dtype).to(device) 
        self.tau_syn = 1. / (1. - self.beta)
        self.alpha = torch.tensor([1 - delta_t / tau_mem], dtype = dtype).to(device) 
        self.tau_mem = 1. / (1. - self.alpha)
        self.gamma = torch.tensor([1 - delta_t / tau_ref], dtype = dtype).to(device)
        self.tau_ref = 1. / (1. - self.gamma)

        
    def state_init(self, batch_size):
        self.P = torch.zeros((batch_size,) + (self.inp_neurons,), dtype = self.dtype).to(self.device)
        self.Q = torch.zeros((batch_size,) + (self.inp_neurons,), dtype = self.dtype).to(self.device)
        self.R = torch.zeros((batch_size,) + (self.out_neurons,), dtype = self.dtype).to(self.device)
        self.S = torch.zeros((batch_size,) + (self.out_neurons,), dtype = self.dtype).to(self.device)
    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R, self.beta * self.Q + input_t

        U = torch.einsum('ab,bc->ac', self.P, self.weights) + self.bias - self.R
        self.S = SmoothStep.apply(U)
        self.R += self.S

        return self.S, U



class SNN_fc_model(torch.nn.Module):
    def __init__(self, x_dim, y_dim, channels, out_classes, tau_ref, tau_mem, tau_syn, delta_t, dtype, device): 
        super(SNN_fc_model, self).__init__()
        self.device = device
        self.dtype  = dtype

        self.FC_layer = LIF_FC_Layer(input_neurons = x_dim*y_dim*channels, output_neurons = 100, tau_syn = tau_syn , tau_mem = tau_mem, tau_ref = tau_ref, delta_t = delta_t, bias = True, device = device, dtype = dtype).to(device)
        self.RO_layer = LIF_FC_Layer(input_neurons = 100, output_neurons = out_classes, tau_syn = tau_syn , tau_mem = tau_mem, tau_ref = tau_ref, delta_t = delta_t, bias = True, device = device, dtype = dtype).to(device)


    def forward(self, inputs):
        # init
        T = inputs.shape[1]
        layer1.state_init(inputs.shape[0])
        layer2.state_init(inputs.shape[0])
        u_rr = torch.zeros((inputs.shape[0], T, args.nclasses), device = self.device)

        # go through time steps
        for t in range(T):
            x              = inputs[:,t,:,:,:].reshape((inputs.shape[0],  -1))
            x, _           = layer1.forward(x)
            _, u_rr[:,t,:] = layer2.forward(x)

        return u_rr


# data loader
train_dl, test_dl = sample_double_mnist_task(
            meta_dataset_type = 'train',
            N = args.nclasses,
            K = args.samples_per_class,
            root='data/nmnist/n_mnist.hdf5',
            batch_size=args.batch_size,
            ds=args.delta_t,
            num_workers=0)

x_preview, y_labels = next(iter(train_dl))
label_to_class = dict(zip(y_labels.unique().tolist(),range(5)))

delta_t = args.delta_t*ms
x_dim   = x_preview.shape[3]
y_dim   = x_preview.shape[4]

tau_mem = torch.tensor([20*ms], dtype = dtype).to(device) #torch.tensor([5*ms, 35*ms], dtype = dtype).to(device)
tau_ref = torch.tensor([1/.35*ms], dtype = dtype).to(device)
tau_syn = torch.tensor([7.5*ms], dtype = dtype).to(device) #torch.tensor([5*ms, 10*ms], dtype = dtype).to(device)

model = SNN_fc_model(x_dim = x_dim, y_dim = y_dim, channels = 2, out_classes = args.nclasses, tau_ref = tau_ref, tau_mem = tau_mem, tau_syn = tau_syn, delta_t = delta_t, dtype = dtype, device = device).to(device)

all_parameters = list(layer1.parameters())
loss_fn = torch.nn.CrossEntropyLoss() 
opt = torch.optim.SGD(all_parameters, lr = args.lr)
acc_hist = []
loss_hist = []

print("Start Training")
print("Epoch   Loss           Accuracy  Time")

# Training
for e in range(args.epochs):
    running_loss = 0
    running_acc  = []
    start_time = time.time()

    for x_data, y_data in train_dl:
        x_data = x_data.to(device)
        y_data = torch.tensor([label_to_class[y.item()] for y in y_data], device = device)

        u_rr = model(x_data)

        #BPTT approach
        loss = loss_fn(u_rr[:,args.burnin:,:].sum(dim = 1), y_data)
        loss.backward()
        opt.step()
        opt.zero_grad()

        # stats
        running_loss += loss.item()
        running_acc.append(u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data)

    acc_hist.append(torch.cat(running_acc).float().mean())
    loss_hist.append(running_loss)
    print("{0:04d}    {1:011.4F}    {2:.4f}    {3:7.4f}".format(e+1, loss_hist[-1], acc_hist[-1], time.time() - start_time))


print("Start Testing")
running_acc  = []
start_time = time.time()
for x_data, y_data in test_dl:
    x_data = x_data.to(device)
    y_data = torch.tensor([label_to_class[y.item()] for y in y_data], device = device)

    u_rr = model(x_data)

    # stats
    running_acc.append(u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data)

print("Test Accuracy: {0:.4f} Test Time {1:7.4f} ".format(torch.cat(running_acc).float().mean(), time.time() - start_time))



