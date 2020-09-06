import torch
import numpy as np

def aux_task_gen(x_data, y_data):
    xtemp = [x_data] 
    ytemp = [y_data, y_data, y_data, y_data]
    aux_ytemp = [ torch.tensor([0]*x_data.shape[0]) ]

    for i in range(1, 4):
        xtemp.append(xtemp[-1].transpose(3,4).flip(1))
        aux_ytemp.append(torch.tensor([i]*x_data.shape[0]))

    return torch.cat(xtemp).to(x_data.device), torch.cat(ytemp).to(x_data.device), torch.cat(aux_ytemp).to(x_data.device)

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
        self.tau_syn = torch.empty(torch.Size((self.inp_neurons,)), dtype = dtype).uniform_(tau_syn_low, tau_syn_high)
        self.beta = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_syn), requires_grad = False)
        self.tau_syn = 1. / (1. - self.beta)

        self.tau_mem = torch.empty(torch.Size((self.inp_neurons,)), dtype = dtype).uniform_(tau_mem_low, tau_mem_high)
        self.alpha = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_mem), requires_grad = False)
        self.tau_mem = 1. / (1. - self.alpha)

        self.tau_ref = torch.empty(torch.Size((self.out_neurons,)), dtype = dtype).uniform_(tau_ref_low, tau_ref_high)
        self.gamma = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_ref), requires_grad = False)
        self.reset = 1. / (1. - self.gamma)

        self.q_mult = self.tau_syn
        self.p_mult = self.tau_mem

        
    def state_init(self, batch_size, device):
        self.P = torch.zeros((batch_size,) + (self.inp_neurons,), dtype = self.dtype, device = device)
        self.Q = torch.zeros((batch_size,) + (self.inp_neurons,), dtype = self.dtype, device = device)
        self.R = torch.zeros((batch_size,) + (self.out_neurons,), dtype = self.dtype, device = device) 
        self.S = torch.zeros((batch_size,) + (self.out_neurons,), dtype = self.dtype, device = device) 
        torch.cuda.empty_cache()
    
    def update_taus(self):
        # make sure tau doesnt dip below... so this is false
        self.beta = torch.clamp(self.beta, min = 0)
        self.tau_syn = 1. / (1. - self.gamma)

        self.alpha = torch.clamp(self.alpha, min = 0)
        self.tau_mem = 1. / (1. - self.alpha)

        self.gamma = torch.clamp(self.gamma, min = 0)
        self.reset = 1. / (1. - self.gamma)

        self.q_mult = self.tau_syn
        self.p_mult = self.tau_mem

    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.p_mult * self.Q, self.gamma * self.R, self.beta * self.Q + self.q_mult * input_t

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
        self.tau_syn = torch.empty(torch.Size((self.inp_dim,)), dtype = dtype).uniform_(tau_syn_low, tau_syn_high)
        self.beta = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_syn), requires_grad = False)
        self.tau_syn = 1. / (1. - self.beta)

        self.tau_mem = torch.empty(torch.Size((self.inp_dim,)), dtype = dtype).uniform_(tau_mem_low, tau_mem_high)
        self.alpha = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_mem), requires_grad = False)
        self.tau_mem = 1. / (1. - self.alpha)

        self.tau_ref = torch.empty(torch.Size((self.out_out,)), dtype = dtype).uniform_(tau_ref_low, tau_ref_high)
        self.gamma = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_ref), requires_grad = False)
        self.reset = 1. / (1. - self.gamma)

        self.q_mult = self.tau_syn
        self.p_mult = self.tau_mem

        

        
    def state_init(self, batch_size, device):
        self.P = torch.zeros((batch_size,) + tuple(self.inp_dim), dtype = self.dtype, device = device)
        self.Q = torch.zeros((batch_size,) + tuple(self.inp_dim), dtype = self.dtype, device = device)
        self.R = torch.zeros((batch_size,) + tuple(self.out_dim), dtype = self.dtype, device = device)
        self.S = torch.zeros((batch_size,) + tuple(self.out_dim), dtype = self.dtype, device = device)
        torch.cuda.empty_cache()
    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.p_mult * self.Q, self.gamma * self.R, self.beta * self.Q + self.q_mult * input_t

        U = self.conv_fwd(self.P) - self.R
        self.S = SmoothStep.apply(U, self.thr)
        self.R += self.S * self.reset

        return self.S, U




class backbone_conv_model(torch.nn.Module):
    def __init__(self, x_preview, in_channels, oc1, oc2, oc3, k1, k2, k3, bias, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, delta_t, reset, thr, gain1, gain2, gain3, dtype): 
        super(backbone_conv_model, self).__init__()
        self.dtype  = dtype

        self.T = x_preview.shape[1]
        x_preview = x_preview[:,0,:,:,:]
        self.mpooling = torch.nn.MaxPool2d(2)

        self.conv_layer1 = LIF_Conv_Layer(x_preview = x_preview, in_channels = in_channels, out_channels = oc1, kernel_size = k1, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, reset = reset, gain = gain1, thr = thr, bias = bias, dtype = dtype)
        x_preview, _ = self.conv_layer1.forward(x_preview)
        x_preview    = self.mpooling(x_preview)

        self.f1_length = x_preview.shape[1] * x_preview.shape[2] * x_preview.shape[3] 

        self.conv_layer2 = LIF_Conv_Layer(x_preview = x_preview, in_channels = oc1, out_channels = oc2, kernel_size = k1, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, reset = reset, gain = gain2, thr = thr, bias = bias, dtype = dtype)
        x_preview, _ = self.conv_layer2.forward(x_preview)
        x_preview    = self.mpooling(x_preview)

        self.f2_length = x_preview.shape[1] * x_preview.shape[2] * x_preview.shape[3] 

        self.conv_layer3 = LIF_Conv_Layer(x_preview = x_preview, in_channels = oc2, out_channels = oc3, kernel_size = k3, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, reset = reset, gain = gain3, thr = thr, bias = bias, dtype = dtype)
        x_preview, _ = self.conv_layer3.forward(x_preview)
        x_preview    = self.mpooling(x_preview)

        self.f_length = x_preview.shape[1] * x_preview.shape[2] * x_preview.shape[3] 


    def forward(self, inputs):
        # init
        self.conv_layer1.state_init(inputs.shape[0], inputs.device)
        self.conv_layer2.state_init(inputs.shape[0], inputs.device)
        self.conv_layer3.state_init(inputs.shape[0], inputs.device)
        s_t = torch.zeros((inputs.shape[0], self.T, self.f_length), device = inputs.device)
        self.spike_count1 = [0] * self.T
        self.spike_count2 = [0] * self.T
        self.spike_count3 = [0] * self.T

        # go through time steps
        for t in range(self.T):
            x, _       = self.conv_layer1.forward(inputs[:,t,:,:,:])
            x          = self.mpooling(x)
            self.spike_count1[t] += x.view(x.shape[0], -1).sum(dim=1).mean().item()
            x, _       = self.conv_layer2.forward(x)
            x          = self.mpooling(x)
            self.spike_count2[t] += x.view(x.shape[0], -1).sum(dim=1).mean().item()
            x, _       = self.conv_layer3.forward(x)
            x          = self.mpooling(x)
            self.spike_count3[t] += x.view(x.shape[0], -1).sum(dim=1).mean().item()
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
