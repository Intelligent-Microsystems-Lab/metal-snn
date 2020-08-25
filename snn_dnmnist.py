import argparse, time, uuid

import torch
import numpy as np
from torchneuromorphic.torchneuromorphic.doublenmnist.doublenmnist_dataloaders import *
from training_curves import plot_curves

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    #torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")
dtype = torch.float32
ms = 1e-3

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=128, help='Batch size')
parser.add_argument("--epochs", type=int, default=10, help='Training Epochs')
parser.add_argument("--epochs-nk", type=int, default=50, help='Training Epochs Few Shot Learning')
parser.add_argument("--burnin", type=int, default=10, help='Burnin Phase in ms')
parser.add_argument("--lr", type=float, default=1.0e-7, help='Learning Rate')
parser.add_argument("--lr-div", type=int, default=100, help='Learning Rate')
parser.add_argument("--init-gain-backbone", type=float, default=.5, help='Gain for weight init') #np.sqrt(2)
parser.add_argument("--init-gain-fc", type=float, default=1, help='Gain for weight init')
parser.add_argument("--n-avg-acc", type=float, default=7, help='Gain for weight init')

parser.add_argument("--train-samples", type=int, default=600, help='Number of samples per classes')
parser.add_argument("--val-samples", type=int, default=600, help='Number of samples per classes')
parser.add_argument("--test-samples", type=int, default=600, help='Number of samples per classes')
parser.add_argument("--aux-classes", type=int, default=4, help='Auxiliar task number of classes (you cant change this)')

parser.add_argument("--n-way", type=int, default=5, help='N-way')
parser.add_argument("--k-shot", type=int, default=1, help='K-shot')

#architecture
parser.add_argument("--k1", type=int, default=7, help='Kernel Size 1')
parser.add_argument("--k2", type=int, default=7, help='Kernel Size 2')
parser.add_argument("--oc1", type=int, default=8, help='Output Channels 1')
parser.add_argument("--oc2", type=int, default=24, help='Output Channels 2')
parser.add_argument("--conv-bias", type=bool, default=True, help='Bias for conv layers')
parser.add_argument("--fc-bias", type=bool, default=True, help='Bias for classifier')

# neural dynamics
parser.add_argument("--delta-t", type=int, default=1, help='Time steps')
parser.add_argument("--tau-mem-low", type=float, default=20, help='Membrane time constant')
parser.add_argument("--tau-syn-low", type=float, default=7.5, help='Synaptic time constant')
parser.add_argument("--tau-ref-low", type=float, default=1/.35, help='Refractory time constant')
parser.add_argument("--tau-mem-high", type=float, default=20, help='Membrane time constant')
parser.add_argument("--tau-syn-high", type=float, default=7.5, help='Synaptic time constant')
parser.add_argument("--tau-ref-high", type=float, default=1/.35, help='Refractory time constant')
parser.add_argument("--reset", type=float, default=1, help='Refractory time constant')
parser.add_argument("--thr", type=float, default=.5, help='Firing Threshold')

args = parser.parse_args()

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''
    @staticmethod
    def forward(aux, x, thr):
        aux.thr = thr
        aux.save_for_backward(x)
        return (x >= thr).float()

    def backward(aux, grad_output):       
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= aux.thr-.5] = 0
        grad_input[input > aux.thr+.5] = 0
        return grad_input, None


class LIF_FC_Layer(torch.nn.Module):
    def __init__(self, input_neurons, output_neurons, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, delta_t, bias, reset, thr, gain, dtype):
        super(LIF_FC_Layer, self).__init__()   
        self.dtype = dtype
        self.inp_neurons = input_neurons      
        self.out_neurons = output_neurons
        self.reset = reset
        self.thr = thr

        self.init =  np.sqrt(6 / (self.inp_neurons)) * gain
                
        self.weights = torch.nn.Parameter(torch.empty((self.inp_neurons, self.out_neurons), dtype = dtype, requires_grad = True))
        torch.nn.init.uniform_(self.weights, a = -self.init, b = self.init)
        #torch.nn.init.xavier_normal_(self.weights, gain = gain)
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((self.out_neurons), dtype = dtype, requires_grad = True))
            torch.nn.init.uniform_(self.bias, a = -0, b = 0)
        else:
            self.bias = None
          

        # taus and betas
        #self.tau_syn = torch.empty(torch.Size((self.inp_neurons,)), dtype = dtype).uniform_(tau_syn_low, tau_syn_high)
        self.beta = torch.nn.Parameter(torch.tensor(1 - delta_t / tau_syn_low), requires_grad = False)

        #self.tau_mem = torch.empty(torch.Size((self.inp_neurons,)), dtype = dtype).uniform_(tau_mem_low, tau_mem_high)
        self.alpha = torch.nn.Parameter(torch.tensor(1 - delta_t / tau_mem_low), requires_grad = False)

        #self.tau_ref = torch.empty(torch.Size((self.out_neurons,)), dtype = dtype).uniform_(tau_ref_low, tau_ref_high)
        self.gamma = torch.nn.Parameter(torch.tensor(1 - delta_t / tau_ref_low), requires_grad = False)

        
    def state_init(self, batch_size, device):
        self.P = torch.zeros((batch_size,) + (self.inp_neurons,), dtype = self.dtype, device = device)
        self.Q = torch.zeros((batch_size,) + (self.inp_neurons,), dtype = self.dtype, device = device)
        self.R = torch.zeros((batch_size,) + (self.out_neurons,), dtype = self.dtype, device = device) 
        self.S = torch.zeros((batch_size,) + (self.out_neurons,), dtype = self.dtype, device = device) 
        torch.cuda.empty_cache()
    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R, self.beta * self.Q + input_t

        U = torch.einsum('ab,bc->ac', self.P, self.weights) + self.bias - self.R
        self.S = SmoothStep.apply(U, self.thr)
        self.R += self.S * self.reset

        return self.S, U

class LIF_Conv_Layer(torch.nn.Module):
    def __init__(self, x_preview, in_channels, out_channels, kernel_size, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, delta_t, bias, thr, reset, gain, dtype):
        super(LIF_Conv_Layer, self).__init__()   
        self.dtype = dtype
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.reset = reset
        self.thr = thr

        self.init =  np.sqrt(6 / ((kernel_size**2) * out_channels )) * gain
        
        self.conv_fwd = torch.nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, kernel_size = self.kernel_size, bias = self.bias)

        torch.nn.init.uniform_(self.conv_fwd.weight, a = -self.init, b = self.init)
        #torch.nn.init.xavier_normal_(self.conv_fwd.weight, gain = gain)
        if bias:
            torch.nn.init.uniform_(self.conv_fwd.bias, a = -0, b = 0)
        else:
            self.bias = None

        self.inp_dim = x_preview.shape[1:]
        self.out_dim = self.conv_fwd(x_preview).shape[1:]
          
        self.state_init(x_preview.shape[0], x_preview.device)

        # taus and betas
        #self.tau_syn = torch.empty(torch.Size(self.inp_dim), dtype = dtype).uniform_(tau_syn_low, tau_syn_high)
        self.beta = torch.nn.Parameter(torch.tensor(1 - delta_t / tau_syn_low), requires_grad = False)

        #self.tau_mem = torch.empty(torch.Size(self.inp_dim), dtype = dtype).uniform_(tau_mem_low, tau_mem_high)
        self.alpha = torch.nn.Parameter(torch.tensor(1 - delta_t / tau_mem_low), requires_grad = False)

        #self.tau_ref = torch.empty(torch.Size(self.out_dim), dtype = dtype).uniform_(tau_ref_low, tau_ref_high)
        self.gamma = torch.nn.Parameter(torch.tensor(1 - delta_t / tau_ref_low), requires_grad = False)

        

        
    def state_init(self, batch_size, device):
        self.P = torch.zeros((batch_size,) + tuple(self.inp_dim), dtype = self.dtype, device = device)
        self.Q = torch.zeros((batch_size,) + tuple(self.inp_dim), dtype = self.dtype, device = device)
        self.R = torch.zeros((batch_size,) + tuple(self.out_dim), dtype = self.dtype, device = device)
        self.S = torch.zeros((batch_size,) + tuple(self.out_dim), dtype = self.dtype, device = device)
        torch.cuda.empty_cache()
    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R, self.beta * self.Q + input_t

        U = self.conv_fwd(self.P) - self.R
        self.S = SmoothStep.apply(U, self.thr)
        self.R += self.S * self.reset

        return self.S, U




class backbone_conv_model(torch.nn.Module):
    def __init__(self, x_preview, in_channels, oc1, oc2, k1, k2, bias, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, delta_t, reset, thr, gain, dtype): 
        super(backbone_conv_model, self).__init__()
        self.device = device
        self.dtype  = dtype

        self.T = x_preview.shape[1]
        x_preview = x_preview[:,0,:,:,:]
        self.mpooling = torch.nn.MaxPool2d(2)

        self.conv_layer1 = LIF_Conv_Layer(x_preview = x_preview, in_channels = in_channels, out_channels = oc1, kernel_size = k1, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, reset = reset, gain = gain, thr = thr, bias = bias, dtype = dtype)
        x_preview, _ = self.conv_layer1.forward(x_preview)
        x_preview    = self.mpooling(x_preview)

        self.f1_length = x_preview.shape[1] * x_preview.shape[2] * x_preview.shape[3] 

        self.conv_layer2 = LIF_Conv_Layer(x_preview = x_preview, in_channels = oc1, out_channels = oc2, kernel_size = k1, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, reset = reset, gain = gain, thr = thr, bias = bias, dtype = dtype)
        x_preview, _ = self.conv_layer2.forward(x_preview)
        x_preview    = self.mpooling(x_preview)

        self.f_length = x_preview.shape[1] * x_preview.shape[2] * x_preview.shape[3] 


    def forward(self, inputs):
        # init
        self.conv_layer1.state_init(inputs.shape[0], inputs.device)
        self.conv_layer2.state_init(inputs.shape[0], inputs.device)
        s_t = torch.zeros((inputs.shape[0], self.T, self.f_length), device = inputs.device)
        self.spike_count1 = [0] * self.T
        self.spike_count2 = [0] * self.T

        # go through time steps
        for t in range(self.T):
            x, _       = self.conv_layer1.forward(inputs[:,t,:,:,:])
            x          = self.mpooling(x)
            self.spike_count1[t] += x.view(x.shape[0], -1).sum(dim=1).mean().item()
            x, _       = self.conv_layer2.forward(x)
            x          = self.mpooling(x)
            self.spike_count2[t] += x.view(x.shape[0], -1).sum(dim=1).mean().item()
            s_t[:,t,:] = x.view(-1,self.f_length)

        return s_t




# classifier
class backbone_fc(torch.nn.Module):
    def __init__(self, T, inp_neurons, output_classes, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, bias, reset, thr, gain, delta_t, dtype): 
        super(backbone_fc, self).__init__()
        self.dtype  = dtype

        self.output_classes = output_classes
        self.T = T

        self.layer1 = LIF_FC_Layer(input_neurons = inp_neurons, output_neurons = output_classes, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, thr = thr, reset = reset, gain = gain, bias = bias, dtype = dtype)
        self.f_length = output_classes


    def forward(self, inputs):
        # init
        self.layer1.state_init(inputs.shape[0], inputs.device)
        s_t = torch.zeros((inputs.shape[0], self.T, self.output_classes), device = inputs.device)
        self.spike_count1 = [0] * self.T

        # go through time steps
        for t in range(self.T):
            s_t[:,t,:], _ = self.layer1.forward(inputs[:,t,:].flatten(start_dim = 1))
            self.spike_count1[t] += s_t[:,t,:].view(s_t[:,t,:].shape[0], -1).sum(dim=1).mean().item()

        return s_t



# classifier
class classifier_model(torch.nn.Module):
    def __init__(self, T, inp_neurons, output_classes, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, bias, reset, thr, gain, delta_t, dtype): 
        super(classifier_model, self).__init__()
        self.dtype  = dtype

        self.output_classes = output_classes
        self.T = T

        self.layer1 = LIF_FC_Layer(input_neurons = inp_neurons, output_neurons = output_classes, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, thr = thr, reset = reset, gain = gain, bias = bias, dtype = dtype)


    def forward(self, inputs): 
        # init
        self.layer1.state_init(inputs.shape[0], inputs.device)
        s_t = torch.zeros((inputs.shape[0], self.T, self.output_classes), device = inputs.device)
        self.spike_count = [0] * self.T

        # go through time steps
        for t in range(self.T):
            s_t[:,t,:], _ = self.layer1.forward(inputs[:,t,:])
            self.spike_count[t] += s_t[:,t,:].view(s_t[:,t,:].shape[0], -1).sum(dim=1).mean().item()

        return s_t

def aux_task_gen(x_data, k):
    y_labels = np.random.choice(k, x_data.shape[0])

    for i in range(1, k):
        c_ind = np.argwhere(y_labels == i)
        x_data[c_ind, :, :, :, :] = torch.rot90(x_data[c_ind, :, :, :, :], i, [4,5])

    return x_data, torch.tensor(y_labels).to(x_data.device)


# data loader
train_dl, _ = sample_double_mnist_task(
            meta_dataset_type = 'train',
            N = args.n_way,
            K = args.train_samples,
            K_test = args.test_samples,
            root='data.nosync/nmnist/n_mnist.hdf5',
            batch_size=args.batch_size,
            ds=args.delta_t,
            num_workers=4)

x_preview, y_labels = next(iter(train_dl))
#label_to_class = dict(zip(y_labels.unique().tolist(),range(args.n_way)))

delta_t = args.delta_t*ms
T = x_preview.shape[1]
 
# backbone Conv
backbone = backbone_conv_model(x_preview = x_preview, in_channels = x_preview.shape[2], oc1 = args.oc1, oc2 = args.oc2, k1 = args.k1, k2 = args.k2, bias = args.conv_bias, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, thr = args.thr, reset = args.reset, gain = args.init_gain_backbone, delta_t = delta_t, dtype = dtype).to(device)

# backbone FC
#backbone = backbone_fc(T = x_preview.shape[1], inp_neurons = 2*64*64, output_classes = 1000, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_backbone, delta_t = delta_t, dtype = dtype).to(device)

classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.nclasses, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)

aux_classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.aux_classes, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)


all_parameters = list(backbone.parameters()) + list(classifier.parameters())
loss_fn = torch.nn.MSELoss(reduction = 'mean')
opt = torch.optim.SGD(all_parameters, lr = args.lr)
acc_hist = []
test_hist = []
aux_hist = []
loss_hist = []
act1_hist = [0]
act2_hist = [0]
act3_hist = [0]
act4_hist = [0]
best_acc = 0
run_e = 0

model_uuid = str(uuid.uuid4())

print(args)
print(model_uuid)
print("Start Training Backbone")
print("Epoch   Loss           Accuracy  S_Conv1   S_Conv2   S_Class   AuxAcc    Time")
for e in range(args.epochs):

    for x_data, y_data in train_dl:
        if run_e%args.lr_div == 0 and (run_e != 0):
            opt.param_groups[-1]['lr'] /= 2
            print("[New lr: {0:6.4f}]".format(opt.param_groups[-1]['lr']))

        start_time = time.time()
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        #y_data = torch.tensor([label_to_class[y.item()] for y in y_data], device = device)

        # create aux task
        x_data, aux_y = aux_task_gen(x_data, args.aux_classes)

        # forwardpass
        bb_rr  = backbone(x_data)
        u_rr   = classifier(bb_rr)
        aux_rr = aux_classifier(bb_rr)
        
        # class loss
        y_onehot = torch.zeros((u_rr.shape[0], u_rr.shape[2]), device = device).scatter_(1,  y_data.unsqueeze(dim = 1), T - args.burnin)
        class_loss = loss_fn(u_rr[:,args.burnin:,:].sum(dim = 1), y_onehot)

        # aux loss
        aux_y_onehot = torch.zeros((aux_rr.shape[0], aux_rr.shape[2]), device = device).scatter_(1,  aux_y.unsqueeze(dim = 1), T - args.burnin)
        aux_loss = loss_fn(aux_rr[:,args.burnin:,:].sum(dim = 1), aux_y_onehot)


        # BPTT
        loss = class_loss + aux_loss
        loss.backward()
        opt.step()
        opt.zero_grad()


        # save stats
        acc_hist.append((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().mean().item())
        aux_hist.append((aux_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().mean().item())
        loss_hist.append(loss.item())
        act1_hist.append(np.sum(backbone.spike_count1[args.burnin:])/(T * backbone.f_length))
        act2_hist.append(np.sum(backbone.spike_count2[args.burnin:])/(T * backbone.f_length))
        act3_hist.append(np.sum(classifier.spike_count[args.burnin:])/(args.nclasses*T))
        
    
        # pring log 
        if run_e%args.n_avg_acc == 0:
            print("{0:04d}    {1:011.4F}    {2:6.4f}    {3:6.4f}    {4:6.4f}    {5:6.4f}    {6:6.4f}    {7:.4f}".format(e+1, loss_hist[-1], np.mean(acc_hist[-args.n_avg_acc:]), act1_hist[-1], act2_hist[-1], act3_hist[-1], np.mean(aux_hist[-args.n_avg_acc:]), time.time() - start_time ))

            # plot train curve
            plot_curves(loss = loss_hist, train = acc_hist, aux = aux_hist, test = test_hist, act1 = act1_hist, act2 = act2_hist, act3 = act3_hist, f_name = model_uuid)
        run_e += 1

    # save model
    checkpoint_dict = {
            'backbone'   : backbone.state_dict(), 
            'classifer'  : classifier.state_dict(),
            'aux_class'  : aux_classifier.state_dict(), 
            'optimizer'  : opt.state_dict(),
            'epoch'      : e, 
            'arguments'  : args,
            'train_loss' : loss_hist,
            'train_curve': acc_hist,
            'aux_curve'  : aux_hist
    }
    torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
    del checkpoint_dict




del classifier, train_dl
torch.cuda.empty_cache()

val_train, _ = sample_double_mnist_task(
            meta_dataset_type = 'val',
            N = args.n_way,
            K = args.val_samples,
            root='data.nosync/nmnist/n_mnist.hdf5',
            batch_size=args.batch_size,
            ds=args.delta_t,
            num_workers=4)

x_pre_val, y_val_labels = next(iter(val_train))
#label_to_val = dict(zip(y_test_labels.unique().tolist(),range(args.n_way)))

ft_classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.ft_nclasses, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)

all_parameters = list(backbone.parameters()) + list(ft_classifier.parameters())
loss_fn = torch.nn.MSELoss(reduction = 'mean')
opt = torch.optim.SGD(all_parameters, lr = args.lr)

print("Start Finetuning Backbone")
print("Epoch   Loss           Accuracy  S_Conv1   S_Conv2   S_Class   AuxAcc    Time")

ft_acc = [0]
best_val = 0
e = 0
while True:
    for x_data, y_data in val_train:
        best_val = np.mean(ft_acc[-args.n_avg_acc:])
        start_time = time.time()
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        #y_data = torch.tensor([label_to_val[y.item()] for y in y_data], device = device)

        # create aux task
        x_data, aux_y = aux_task_gen(x_data, args.aux_classes)

        # forwardpass
        bb_rr  = backbone(x_data)
        u_rr   = ft_classifier(bb_rr)
        aux_rr = aux_classifier(bb_rr)
        
        # class loss
        y_onehot = torch.zeros((u_rr.shape[0], u_rr.shape[2]), device = device).scatter_(1,  y_data.unsqueeze(dim = 1), T - args.burnin)
        class_loss = loss_fn(u_rr[:,args.burnin:,:].sum(dim = 1), y_onehot)

        # aux loss
        aux_y_onehot = torch.zeros((aux_rr.shape[0], aux_rr.shape[2]), device = device).scatter_(1,  aux_y.unsqueeze(dim = 1), T - args.burnin)
        aux_loss = loss_fn(aux_rr[:,args.burnin:,:].sum(dim = 1), aux_y_onehot)

        # Manifold loss...

        # BPTT
        loss = class_loss + aux_loss
        loss.backward()
        opt.step()
        opt.zero_grad()

        # save stats
        acc_hist.append((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().mean().item())
        ft_acc.append(acc_hist[-1])
        aux_hist.append((aux_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().mean().item())
        loss_hist.append(loss.item())
        act1_hist.append(np.sum(backbone.spike_count1[args.burnin:])/(T * backbone.f_length))
        act2_hist.append(np.sum(backbone.spike_count2[args.burnin:])/(T * backbone.f_length))
        act3_hist.append(np.sum(ft_classifier.spike_count[args.burnin:])/(args.nclasses*T))
    
        # pring log 
        print("{0:04d}    {1:011.4F}    {2:6.4f}    {3:6.4f}    {4:6.4f}    {5:6.4f}    {6:6.4f}    {7:.4f}".format(e+1, loss_hist[-1], np.mean(ft_acc[-args.n_avg_acc:]), act1_hist[-1], act2_hist[-1], act3_hist[-1], np.mean(aux_hist[-args.n_avg_acc:]), time.time() - start_time ))
        # plot train curve
        plot_curves(loss = loss_hist, train = acc_hist, aux = aux_hist, test = test_hist, act1 = act1_hist, act2 = act2_hist, act3 = act3_hist, f_name = model_uuid)
        if best_val > np.mean(ft_acc[-args.n_avg_acc:]):
            break
    else:
        continue
    break

    e += 1
    # save model
    checkpoint_dict = {
            'backbone'     : backbone.state_dict(), 
            #'classifer'    : classifier.state_dict(),
            'ft_classifer' : ft_classifier.state_dict(),
            'aux_class'    : aux_classifier.state_dict(), 
            'optimizer'    : opt.state_dict(),
            'epoch'        : e, 
            'arguments'    : args,
            'train_loss'   : loss_hist,
            'train_curve'  : acc_hist,
            'aux_curve'    : aux_hist
    }
    torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
    del checkpoint_dict

del ft_classifier, aux_classifier, val_train
torch.cuda.empty_cache()

nk_train, nk_test = sample_double_mnist_task(
            meta_dataset_type = 'test',
            N = args.n_way,
            K = args.k_shot,
            K_test = args.test_samples
            root='data.nosync/nmnist/n_mnist.hdf5',
            batch_size=args.batch_size,
            ds=args.delta_t,
            num_workers=4)

nk_classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.n_way, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)

x_pre_test, y_test_labels = next(iter(nk_train))
#label_to_test = dict(zip(y_test_labels.unique().tolist(),range(args.n_way)))

# Final Test of few shot learning
loss_fn = torch.nn.MSELoss(reduction = 'mean')
opt = torch.optim.SGD(nk_classifier.parameters(), lr = args.lr)

print("Start Few Shot Learning")
print("Epoch   Loss           Accuracy  S_Conv1   S_Conv2   S_Class   Time")
best_val = 0
final_acc = [0]
for e in range(args.epochs_nk):
    for x_data, y_data in nk_train:
        start_time = time.time()
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        #y_data = torch.tensor([label_to_test[y.item()] for y in y_data], device = device)

        # forwardpass
        bb_rr  = backbone(x_data)
        u_rr   = nk_classifier(bb_rr)
        
        # class loss
        y_onehot = torch.zeros((u_rr.shape[0], u_rr.shape[2]), device = device).scatter_(1,  y_data.unsqueeze(dim = 1), T - args.burnin)
        class_loss = loss_fn(u_rr[:,args.burnin:,:].sum(dim = 1), y_onehot)

        # BPTT
        class_loss.backward()
        opt.step()
        opt.zero_grad()

        # save stats
        acc_hist.append((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().mean().item())
        final_acc.append(acc_hist[-1])
        loss_hist.append(loss.item())
        act1_hist.append(np.sum(backbone.spike_count1[args.burnin:])/(T * backbone.f_length))
        act2_hist.append(np.sum(backbone.spike_count2[args.burnin:])/(T * backbone.f_length))
        act3_hist.append(np.sum(nk_classifier.spike_count[args.burnin:])/(args.nclasses*T))
    
        # pring log 
        print("{0:04d}    {1:011.4F}    {2:6.4f}    {3:6.4f}    {4:6.4f}    {5:6.4f}    {6:.4f}".format(e+1, loss_hist[-1], final_acc[-1], act1_hist[-1], act2_hist[-1], act3_hist[-1], time.time() - start_time ))
        # plot train curve
        plot_curves(loss = loss_hist, train = acc_hist, aux = aux_hist, test = test_hist, act1 = act1_hist, act2 = act2_hist, act3 = act3_hist, f_name = model_uuid)

    # save model
    checkpoint_dict = {
            'backbone'     : backbone.state_dict(), 
            #'classifer'    : classifier.state_dict(),
            #'ft_classifer' : ft_classifier.state_dict(),
            'nk_classifer' : nk_classifier.state_dict(),
            #'aux_class'    : aux_classifier.state_dict(), 
            'optimizer'    : opt.state_dict(),
            'epoch'        : e, 
            'arguments'    : args,
            'train_loss'   : loss_hist,
            'train_curve'  : acc_hist,
            'aux_curve'    : aux_hist
    }
    torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
    del checkpoint_dict


