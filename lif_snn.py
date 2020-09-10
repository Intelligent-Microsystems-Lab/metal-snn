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
    def __init__(self, input_neurons, output_neurons, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, delta_t, bias, reset, thr, gain, train_t, dtype):
        super(LIF_FC_Layer, self).__init__()   
        self.dtype = dtype
        self.inp_neurons = input_neurons      
        self.out_neurons = output_neurons
        self.thr = thr
        self.train_t = train_t

        #self.init =  np.sqrt(6 / (self.inp_neurons)) * gain
        self.init = gain

        self.weights = torch.nn.Parameter(torch.empty((self.inp_neurons, self.out_neurons), dtype = dtype, requires_grad = True))
        torch.nn.init.uniform_(self.weights, a = -self.init, b = self.init)
        #torch.nn.init.xavier_normal_(self.weights, gain = gain)
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((self.out_neurons), dtype = dtype, requires_grad = True))
            torch.nn.init.uniform_(self.bias, a = -0, b = 0)
        else:
            self.bias = None
          
        # taus and betas
        if tau_syn_high == tau_syn_high:
            self.beta = torch.tensor(1 - delta_t / tau_syn_high)
            self.tau_syn = torch.tensor(1. / (1. - self.beta))
            self.q_mult = self.tau_syn
        else:
            self.beta_high = 1 - delta_t / tau_syn_high
            self.beta_low = 1 - delta_t / tau_syn_low
            self.tau_syn = torch.empty(input_neurons, dtype = dtype).uniform_(tau_syn_low, tau_syn_high)
            self.beta = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_syn), requires_grad = train_t)
            self.tau_syn = torch.nn.Parameter(1. / (1. - self.beta), requires_grad = False)
            self.q_mult = torch.nn.Parameter(self.tau_syn, requires_grad = False)

        if tau_mem_high == tau_mem_low:
            self.alpha = torch.tensor(1 - delta_t / tau_mem_high)
            self.tau_mem = torch.tensor(1. / (1. - self.alpha))
            self.p_mult = self.tau_mem
        else:
            self.alpha_high = 1 - delta_t / tau_mem_high
            self.alpha_low = 1 - delta_t / tau_mem_low
            self.tau_mem = torch.empty(input_neurons, dtype = dtype).uniform_(tau_mem_low, tau_mem_high)
            self.alpha = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_mem), requires_grad = train_t)
            self.tau_mem = torch.nn.Parameter(1. / (1. - self.alpha), requires_grad = False)
            self.p_mult = torch.nn.Parameter(self.tau_mem, requires_grad = False)

        if tau_ref_high == tau_ref_low:
            self.gamma = torch.tensor(1 - delta_t / tau_ref_high)
            self.reset = torch.tensor(1. / (1. - self.gamma))
            self.r_mult = self.reset
        else:
            self.gamma_high = 1 - delta_t / tau_ref_high
            self.gamma_low = 1 - delta_t / tau_ref_low
            self.tau_ref = torch.empty(output_neurons, dtype = dtype).uniform_(tau_ref_low, tau_ref_high)
            self.gamma = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_ref), requires_grad = train_t)
            self.reset = torch.nn.Parameter(1. / (1. - self.gamma), requires_grad = False)
            self.r_mult = torch.nn.Parameter(self.reset, requires_grad = False)
        
    def state_init(self, batch_size, device):
        self.P = torch.zeros((batch_size,) + (self.inp_neurons,), dtype = self.dtype, device = device)
        self.Q = torch.zeros((batch_size,) + (self.inp_neurons,), dtype = self.dtype, device = device)
        self.R = torch.zeros((batch_size,) + (self.out_neurons,), dtype = self.dtype, device = device) 
        self.S = torch.zeros((batch_size,) + (self.out_neurons,), dtype = self.dtype, device = device) 
        torch.cuda.empty_cache()
    
    # retired - until time constant learning reintroduce
    # def update_taus(self):
    #     self.beta = torch.nn.Parameter(torch.clamp(self.beta, max = self.beta_high, min = self.beta_low), requires_grad = self.train_t).to(self.P.device)
    #     self.tau_syn = 1. / (1. - self.beta)

    #     self.alpha = torch.nn.Parameter(torch.clamp(self.alpha, max = self.alpha_high, min = self.alpha_low), requires_grad = self.train_t).to(self.P.device)
    #     self.tau_mem = 1. / (1. - self.alpha)

    #     self.gamma = torch.nn.Parameter(torch.clamp(self.gamma, max = self.gamma_high, min = self.gamma_low), requires_grad = self.train_t).to(self.P.device)
    #     self.reset = 1. / (1. - self.gamma)

    #     self.q_mult = torch.nn.Parameter(self.tau_syn, requires_grad = False).to((self.P.device))
    #     self.p_mult = torch.nn.Parameter(self.tau_mem, requires_grad = False).to((self.P.device))
    #     self.r_mult = torch.nn.Parameter(self.reset, requires_grad = False).to((self.P.device))

    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.p_mult * self.Q, self.gamma * self.R, self.beta * self.Q + self.q_mult * input_t

        U = torch.einsum('ab,bc->ac', self.P, self.weights) + self.bias - self.R
        self.S = SmoothStep.apply(U, self.thr)
        self.R += self.S * self.r_mult

        return self.S, U


class LIF_Conv_Layer(torch.nn.Module):
    def __init__(self, x_preview, in_channels, out_channels, kernel_size, padding, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, delta_t, bias, thr, reset, gain, train_t, dtype):
        super(LIF_Conv_Layer, self).__init__()   
        self.dtype = dtype
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.reset = reset
        self.thr = thr
        self.train_t = train_t

        #self.init =  np.sqrt(6 / ((kernel_size**2) * out_channels )) * gain
        self.init = gain

        self.conv_fwd = torch.nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, kernel_size = self.kernel_size, bias = self.bias, padding = padding)

        torch.nn.init.uniform_(self.conv_fwd.weight, a = -self.init, b = self.init)
        #torch.nn.init.xavier_normal_(self.conv_fwd.weight, gain = gain)
        if bias:
            torch.nn.init.uniform_(self.conv_fwd.bias, a = -0, b = 0)
        else:
            self.bias = None

        self.inp_dim = [int(x) for x in x_preview.shape[1:]]
        self.out_dim = [int(x) for x in self.conv_fwd(x_preview).shape[1:]]
          
        self.state_init(x_preview.shape[0], x_preview.device)

        # taus and betas
        if tau_syn_high == tau_syn_high:
            self.beta = torch.tensor(1 - delta_t / tau_syn_high)
            self.tau_syn = torch.tensor(1. / (1. - self.beta))
            self.q_mult = self.tau_syn
        else:
            self.beta_high = 1 - delta_t / tau_syn_high
            self.beta_low = 1 - delta_t / tau_syn_low
            self.tau_syn = torch.empty(input_neurons, dtype = dtype).uniform_(tau_syn_low, tau_syn_high)
            self.beta = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_syn), requires_grad = train_t)
            self.tau_syn = torch.nn.Parameter(1. / (1. - self.beta), requires_grad = False)
            self.q_mult = torch.nn.Parameter(self.tau_syn, requires_grad = False)

        if tau_mem_high == tau_mem_low:
            self.alpha = torch.tensor(1 - delta_t / tau_mem_high)
            self.tau_mem = torch.tensor(1. / (1. - self.alpha))
            self.p_mult = self.tau_mem
        else:
            self.alpha_high = 1 - delta_t / tau_mem_high
            self.alpha_low = 1 - delta_t / tau_mem_low
            self.tau_mem = torch.empty(input_neurons, dtype = dtype).uniform_(tau_mem_low, tau_mem_high)
            self.alpha = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_mem), requires_grad = train_t)
            self.tau_mem = torch.nn.Parameter(1. / (1. - self.alpha), requires_grad = False)
            self.p_mult = torch.nn.Parameter(self.tau_mem, requires_grad = False)

        if tau_ref_high == tau_ref_low:
            self.gamma = torch.tensor(1 - delta_t / tau_ref_high)
            self.reset = torch.tensor(1. / (1. - self.gamma))
            self.r_mult = self.reset
        else:
            self.gamma_high = 1 - delta_t / tau_ref_high
            self.gamma_low = 1 - delta_t / tau_ref_low
            self.tau_ref = torch.empty(output_neurons, dtype = dtype).uniform_(tau_ref_low, tau_ref_high)
            self.gamma = torch.nn.Parameter(torch.tensor(1 - delta_t / self.tau_ref), requires_grad = train_t)
            self.reset = torch.nn.Parameter(1. / (1. - self.gamma), requires_grad = False)
            self.r_mult = torch.nn.Parameter(self.reset, requires_grad = False)
        

        
    def state_init(self, batch_size, device):
        self.P = torch.zeros((batch_size,) + tuple(self.inp_dim), dtype = self.dtype, device = device)
        self.Q = torch.zeros((batch_size,) + tuple(self.inp_dim), dtype = self.dtype, device = device)
        self.R = torch.zeros((batch_size,) + tuple(self.out_dim), dtype = self.dtype, device = device)
        self.S = torch.zeros((batch_size,) + tuple(self.out_dim), dtype = self.dtype, device = device)
        torch.cuda.empty_cache()

    # retired - until time constant learning reintroduce
    # def update_taus(self):
    #     self.beta = torch.nn.Parameter(torch.clamp(self.beta, max = self.beta_high, min = self.beta_low), requires_grad = self.train_t).to(self.P.device)
    #     self.tau_syn = 1. / (1. - self.beta)

    #     self.alpha = torch.nn.Parameter(torch.clamp(self.alpha, max = self.alpha_high, min = self.alpha_low), requires_grad = self.train_t).to(self.P.device)
    #     self.tau_mem = 1. / (1. - self.alpha)

    #     self.gamma = torch.nn.Parameter(torch.clamp(self.gamma, max = self.gamma_high, min = self.gamma_low), requires_grad = self.train_t).to(self.P.device)
    #     self.reset = 1. / (1. - self.gamma)

    #     self.q_mult = torch.nn.Parameter(self.tau_syn, requires_grad = False).to((self.P.device))
    #     self.p_mult = torch.nn.Parameter(self.tau_mem, requires_grad = False).to((self.P.device))
    #     self.r_mult = torch.nn.Parameter(self.reset, requires_grad = False).to((self.P.device))
    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.p_mult * self.Q, self.gamma * self.R, self.beta * self.Q + self.q_mult * input_t

        U = self.conv_fwd(self.P) - self.R
        self.S = SmoothStep.apply(U, self.thr)
        self.R += self.S * self.r_mult

        return self.S, U




class backbone_conv_model(torch.nn.Module):
    def __init__(self, x_preview, in_channels, oc1, oc2, oc3, k1, k2, k3, padding, bias, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, delta_t, reset, thr, gain1, gain2, gain3, train_t, dtype): 
        super(backbone_conv_model, self).__init__()
        self.dtype  = dtype

        self.T = int(x_preview.shape[1])
        x_preview = x_preview[:,0,:,:,:]
        self.mpooling = torch.nn.MaxPool2d(2)

        self.conv_layer1 = LIF_Conv_Layer(x_preview = x_preview, in_channels = in_channels, out_channels = oc1, kernel_size = k1, padding = padding, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, reset = reset, gain = gain1, thr = thr, bias = bias, train_t = train_t, dtype = dtype)
        x_preview, _ = self.conv_layer1.forward(x_preview)
        x_preview    = self.mpooling(x_preview)

        self.f1_length = x_preview.shape[1] * x_preview.shape[2] * x_preview.shape[3] 

        self.conv_layer2 = LIF_Conv_Layer(x_preview = x_preview, in_channels = oc1, out_channels = oc2, kernel_size = k1, padding = padding, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, reset = reset, gain = gain2, thr = thr, bias = bias, train_t = train_t, dtype = dtype)
        x_preview, _ = self.conv_layer2.forward(x_preview)
        #x_preview    = self.mpooling(x_preview)

        self.f2_length = x_preview.shape[1] * x_preview.shape[2] * x_preview.shape[3] 

        self.conv_layer3 = LIF_Conv_Layer(x_preview = x_preview, in_channels = oc2, out_channels = oc3, kernel_size = k3, padding = padding, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, reset = reset, gain = gain3, thr = thr, bias = bias, train_t = train_t, dtype = dtype)
        x_preview, _ = self.conv_layer3.forward(x_preview)
        x_preview    = self.mpooling(x_preview)

        self.f_length = x_preview.shape[1] * x_preview.shape[2] * x_preview.shape[3] 
        del x_preview

    def update_taus(self):
        self.conv_layer1.update_taus()
        self.conv_layer2.update_taus()
        self.conv_layer3.update_taus()


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
            self.spike_count1[t] += int(x.view(x.shape[0], -1).sum(dim=1).mean().item())
            x, _       = self.conv_layer2.forward(x)
            #x          = self.mpooling(x)
            self.spike_count2[t] += int(x.view(x.shape[0], -1).sum(dim=1).mean().item())
            x, _       = self.conv_layer3.forward(x)
            x          = self.mpooling(x)
            self.spike_count3[t] += int(x.view(x.shape[0], -1).sum(dim=1).mean().item())
            s_t[:,t,:] = x.view(-1,self.f_length)

        return s_t



# retired
# # classifier
# class backbone_fc(torch.nn.Module):
#     def __init__(self, T, inp_neurons, output_classes, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, bias, reset, thr, gain, delta_t, train_t, dtype): 
#         super(backbone_fc, self).__init__()
#         self.dtype  = dtype

#         self.output_classes = output_classes
#         self.T = T

#         self.layer1 = LIF_FC_Layer(input_neurons = inp_neurons, output_neurons = output_classes, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, thr = thr, reset = reset, gain = gain, bias = bias, dtype = dtype)
#         self.f_length = output_classes

#     def update_taus(self):
#         self.layer1.update_taus()

#     def forward(self, inputs):
#         # init
#         self.layer1.state_init(inputs.shape[0], inputs.device)
#         s_t = torch.zeros((inputs.shape[0], self.T, self.output_classes), device = inputs.device)
#         self.spike_count1 = [0] * self.T

#         # go through time steps
#         for t in range(self.T):
#             s_t[:,t,:], _ = self.layer1.forward(inputs[:,t,:].flatten(start_dim = 1))
#             self.spike_count1[t] += s_t[:,t,:].view(s_t[:,t,:].shape[0], -1).sum(dim=1).mean().item()

#         return s_t



# classifier
class classifier_model(torch.nn.Module):
    def __init__(self, T, inp_neurons, output_classes, tau_syn_low, tau_mem_low, tau_ref_low, tau_syn_high, tau_mem_high, tau_ref_high, bias, reset, thr, gain, delta_t, train_t, dtype): 
        super(classifier_model, self).__init__()
        self.dtype  = dtype

        self.output_classes = output_classes
        self.T = T

        self.layer1 = LIF_FC_Layer(input_neurons = inp_neurons, output_neurons = output_classes, tau_syn_low = tau_syn_low, tau_mem_low = tau_mem_low, tau_ref_low = tau_ref_low, tau_syn_high = tau_syn_high, tau_mem_high = tau_mem_high, tau_ref_high = tau_ref_high, delta_t = delta_t, thr = thr, reset = reset, gain = gain, bias = bias, train_t = train_t, dtype = dtype)

    def update_taus(self):
        self.layer1.update_taus()

    def forward(self, inputs): 
        # init
        self.layer1.state_init(inputs.shape[0], inputs.device)
        s_t = torch.zeros((inputs.shape[0], self.T, self.output_classes), device = inputs.device)
        self.spike_count = [0] * self.T

        # go through time steps
        for t in range(self.T):
            s_t[:,t,:], _ = self.layer1.forward(inputs[:,t,:])
            self.spike_count[t] += int(s_t[:,t,:].view(s_t[:,t,:].shape[0], -1).sum(dim=1).mean().item())

        return s_t
