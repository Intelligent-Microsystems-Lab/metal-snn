import argparse, time, uuid

import torch
import numpy as np
from torchneuromorphic.torchneuromorphic.doublenmnist.doublenmnist_dataloaders import *
from training_curves import plot_curves

from lif_snn import backbone_conv_model, classifier_model#, aux_task_gen

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    #torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")
dtype = torch.float32
ms = 1e-3

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=16, help='Batch size')
parser.add_argument("--batch-size-test", type=int, default=4, help='Batch size')
parser.add_argument("--batch-size-test-test", type=int, default=128, help='Batch size test test')
parser.add_argument("--epochs", type=int, default=400, help='Training Epochs')
parser.add_argument("--epochs-nk", type=int, default=301, help='Training Epochs Few Shot Learning')
parser.add_argument("--iter-test", type=int, default=600, help='Test Iter')
parser.add_argument("--burnin", type=int, default=10, help='Burnin Phase in ms')
parser.add_argument("--lr", type=float, default=1.0e-7, help='Learning Rate')
#parser.add_argument("--lr-div", type=int, default=100, help='Learning Rate')
parser.add_argument("--init-gain-backbone", type=float, default=.5, help='Gain for weight init') #np.sqrt(2)
parser.add_argument("--init-gain-fc", type=float, default=1, help='Gain for weight init')
#parser.add_argument("--log-int", type=int, default=5, help='Logging Interval')
#parser.add_argument("--n-avg-acc", type=int, default=10, help='Averaging for Acc')

parser.add_argument("--train-samples", type=int, default=100, help='Number of samples per classes')
parser.add_argument("--val-samples", type=int, default=100, help='Number of samples per classes')
parser.add_argument("--test-samples", type=int, default=100, help='Number of samples per classes')
parser.add_argument("--aux-classes", type=int, default=4, help='Auxiliar task number of classes (you cant change this)')
#parser.add_argument("--aux-impo", type=float, default=.5, help='Weight of aux loss')

parser.add_argument("--n-train", type=int, default=20, help='N-way for training technically I guess more (?)')

parser.add_argument("--n-way", type=int, default=5, help='N-way')
parser.add_argument("--k-shot", type=int, default=5, help='K-shot')

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
parser.add_argument("--target_act", type=float, default=.95, help='Firing Threshold')
parser.add_argument("--none_act", type=float, default=.05, help='Firing Threshold')

args = parser.parse_args()


def aux_task_gen(x_data, k, y_data):
    xtemp = [x_data] 
    ytemp = [y_data, y_data, y_data, y_data]
    aux_ytemp = [ torch.tensor([0]*x_data.shape[0]) ]

    for i in range(1, k):
        xtemp.append(xtemp[-1].transpose(3,4).flip(1))
        aux_ytemp.append(torch.tensor([i]*x_data.shape[0]))

    return torch.cat(xtemp).to(x_data.device), torch.cat(ytemp).to(x_data.device), torch.cat(aux_ytemp).to(x_data.device)

# training data
train_dl, test_dl  = sample_double_mnist_task(
            meta_dataset_type = 'train',
            N = args.n_train,
            K = args.train_samples,
            K_test = args.test_samples,
            root='data.nosync/nmnist/n_mnist.hdf5',
            batch_size=args.batch_size,
            batch_size_test=args.batch_size,
            ds=args.delta_t,
            num_workers=4)
x_preview, y_labels = next(iter(train_dl))

delta_t = args.delta_t*ms
T = x_preview.shape[1]
max_act = T - args.burnin
 
# backbone Conv
backbone = backbone_conv_model(x_preview = x_preview, in_channels = x_preview.shape[2], oc1 = args.oc1, oc2 = args.oc2, k1 = args.k1, k2 = args.k2, bias = args.conv_bias, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, thr = args.thr, reset = args.reset, gain = args.init_gain_backbone, delta_t = delta_t, dtype = dtype).to(device)

classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.n_train, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)

aux_classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.aux_classes, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)



loss_fn = torch.nn.MSELoss(reduction = 'mean')
opt = torch.optim.Adam([
                {'params': backbone.parameters()},
                {'params': classifier.parameters()},
                {'params': aux_classifier.parameters()}
            ], lr = args.lr) #SGD before


acc_hist = []
test_hist = []
aux_hist = []
loss_hist = []
act1_hist = [0]
act2_hist = [0]
act3_hist = [0]
act4_hist = [0]
best_acc = 0

model_uuid = str(uuid.uuid4())

print(args)
print(model_uuid)
print("Start Training Backbone")

for e in range(args.epochs):
    avg_loss=0
    avg_rloss=0

    for i, (x_data, y_data) in enumerate(train_dl):
        start_time = time.time()
        x_data = x_data.to(device)

        # create aux task
        x_data, y_data, aux_y  = aux_task_gen(x_data, args.aux_classes, y_data)

        # forwardpass
        bb_rr  = backbone(x_data)
        u_rr   = classifier(bb_rr)
        aux_rr = aux_classifier(bb_rr)
        
        # class loss
        y_onehot = torch.zeros((u_rr.shape[0], u_rr.shape[2]), device = device).scatter_(1,  y_data.unsqueeze(dim = 1), (max_act*args.target_act) - (max_act*args.none_act)) + (max_act*args.none_act)
        class_loss = loss_fn(u_rr[:,args.burnin:,:].sum(dim = 1), y_onehot)

        # aux loss
        aux_y_onehot = torch.zeros((aux_rr.shape[0], aux_rr.shape[2]), device = device).scatter_(1,  aux_y.unsqueeze(dim = 1), (max_act*args.target_act) - (max_act*args.none_act)) + (max_act*args.none_act)
        aux_loss = loss_fn(aux_rr[:,args.burnin:,:].sum(dim = 1), aux_y_onehot)


        # BPTT
        loss = .5 * class_loss + .5 * aux_loss
        loss.backward()
        opt.step()
        opt.zero_grad()

        avg_loss = avg_loss+class_loss.data.item()
        avg_rloss = avg_rloss+aux_loss.data.item()

        if i % 10 == 0:
            print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Rotate Loss {:f} | Time {:f}'.format(e+1, i, len(train_dl), avg_loss/float(i+1), avg_rloss/float(i+1), time.time() - start_time ))
        

    # then accuracy on test!
    with torch.no_grad():
        correct = rcorrect = total = 0
        for x_data, y_data in test_dl:

            start_time = time.time()
            x_data = x_data.to(device)

            # create aux task
            x_data, y_data, aux_y  = aux_task_gen(x_data, args.aux_classes, y_data)

            # forwardpass
            bb_rr  = backbone(x_data)
            u_rr   = classifier(bb_rr)
            aux_rr = aux_classifier(bb_rr)
            
            correct += (u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().sum()
            rcorrect += (aux_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().sum()
            total += x_data.shape[0]
    torch.cuda.empty_cache()

    print("Epoch {:d} : Accuracy {:f}, Rotate Accuracy {:f}".format(e+1,(float(correct)*100)/total,(float(rcorrect)*100)/total))

    # model save!


del train_dl, classifier
torch.cuda.empty_cache()

# val_train, _ = sample_double_mnist_task(
#             meta_dataset_type = 'val',
#             N = args.n_way,
#             K = args.val_samples,
#             K_test = args.val_samples,
#             root='data.nosync/nmnist/n_mnist.hdf5',
#             batch_size=args.batch_size,
#             ds=args.delta_t,
#             num_workers=4)

# x_pre_val, y_val_labels = next(iter(val_train))

# all_parameters = list(backbone.parameters()) + list(classifier.parameters())
# loss_fn = torch.nn.MSELoss(reduction = 'mean')
# opt = torch.optim.SGD(all_parameters, lr = args.lr)

# print("Start Finetuning Backbone")
# print("Epoch   Loss           Accuracy  S_Conv1   S_Conv2   S_Class   AuxAcc    Time")

# ft_acc = [0]
# best_val = 0
# run_e = 0
# while True:
#     for x_data, y_data in val_train:
#         best_val = np.mean(ft_acc[-args.n_avg_acc:])
#         start_time = time.time()
#         x_data = x_data.to(device)
#         y_data = y_data.to(device)
#         #y_data = torch.tensor([label_to_val[y.item()] for y in y_data], device = device)

#         # create aux task
#         x_data, aux_y = aux_task_gen(x_data, args.aux_classes)

#         # forwardpass
#         bb_rr  = backbone(x_data)
#         u_rr   = classifier(bb_rr)
#         aux_rr = aux_classifier(bb_rr)
        
#         # class loss
#         y_onehot = torch.zeros((u_rr.shape[0], u_rr.shape[2]), device = device).scatter_(1,  y_data.unsqueeze(dim = 1), (max_act*args.target_act) - (max_act*args.none_act)) + (max_act*args.none_act)
#         class_loss = loss_fn(u_rr[:,args.burnin:,:].sum(dim = 1), y_onehot)

#         # aux loss
#         aux_y_onehot = torch.zeros((aux_rr.shape[0], aux_rr.shape[2]), device = device).scatter_(1,  aux_y.unsqueeze(dim = 1), T - args.burnin)
#         aux_loss = loss_fn(aux_rr[:,args.burnin:,:].sum(dim = 1), aux_y_onehot)

#         # Manifold loss...

#         # BPTT
#         loss = class_loss + args.aux_impo * aux_loss
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#         # save stats
#         acc_hist.append((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().mean().item())
#         ft_acc.append(acc_hist[-1])
#         aux_hist.append((aux_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().mean().item())
#         loss_hist.append(loss.item())
#         act1_hist.append(np.sum(backbone.spike_count1[args.burnin:])/(T * backbone.f_length))
#         act2_hist.append(np.sum(backbone.spike_count2[args.burnin:])/(T * backbone.f_length))
#         act3_hist.append(np.sum(classifier.spike_count[args.burnin:])/(args.n_way*T))
    
#         # pring log 
#         print("{0:04d}    {1:011.4F}    {2:6.4f}    {3:6.4f}    {4:6.4f}    {5:6.4f}    {6:6.4f}    {7:.4f}".format(run_e+1, loss_hist[-1], np.mean(ft_acc[-args.n_avg_acc:]), act1_hist[-1], act2_hist[-1], act3_hist[-1], np.mean(aux_hist[-args.n_avg_acc:]), time.time() - start_time ))
#         # plot train curve
#         plot_curves(loss = loss_hist, train = acc_hist, aux = aux_hist, test = test_hist, act1 = act1_hist, act2 = act2_hist, act3 = act3_hist, f_name = model_uuid)
#         if best_val > np.mean(ft_acc[-args.n_avg_acc:]):
#             break
#     else:
#         # save model
#         checkpoint_dict = {
#                 'backbone'     : backbone.state_dict(), 
#                 'classifer'    : classifier.state_dict(),
#                 'aux_class'    : aux_classifier.state_dict(), 
#                 'optimizer'    : opt.state_dict(),
#                 'epoch'        : e, 
#                 'arguments'    : args,
#                 'train_loss'   : loss_hist,
#                 'train_curve'  : acc_hist,
#                 'aux_curve'    : aux_hist
#         }
#         torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
#         del checkpoint_dict
#         continue
#     break

    

# del val_train, aux_classifier
# torch.cuda.empty_cache()


##################
# META Training
##################

print("Evaluating over %d classes with %d examples"%(args.n_way, args.k_shot))

acc_all = [[],[],[]]
for i in range(args.iter_test):

    # new task
    support_ds, query_ds = sample_double_mnist_task(
                meta_dataset_type = 'test',
                N = args.n_way,
                K = args.k_shot,
                K_test = args.test_samples,
                root='data.nosync/nmnist/n_mnist.hdf5',
                batch_size = args.batch_size_test,
                batch_size_test = args.batch_size_test_test,
                ds=args.delta_t,
                num_workers=4)

    x_preview, _ = next(iter(support_ds))

    delta_t = args.delta_t*ms
    T = x_preview.shape[1]
    max_act = T - args.burnin

    classifier_nk = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.n_way, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)

    # Final Test of few shot learning
    loss_fn = torch.nn.MSELoss(reduction = 'mean')
    opt = torch.optim.Adam(classifier_nk.parameters(), lr = args.lr) # SGD before

    for e in range(args.epochs_nk):
        print(".", end='')
        for x_data, y_data in support_ds:
            start_time = time.time()
            x_data = x_data.to(device)
            y_data = y_data.to(device)

            # forwardpass
            bb_rr  = backbone(x_data)
            u_rr   = classifier_nk(bb_rr)
            
            # class loss
            y_onehot = torch.zeros((u_rr.shape[0], u_rr.shape[2]), device = device).scatter_(1,  y_data.unsqueeze(dim = 1), (max_act*args.target_act) - (max_act*args.none_act)) + (max_act*args.none_act)
            class_loss = loss_fn(u_rr[:,args.burnin:,:].sum(dim = 1), y_onehot)

            # BPTT
            class_loss.backward()
            opt.step()
            opt.zero_grad()

        del x_data, y_data, y_onehot, bb_rr, u_rr, class_loss
        torch.cuda.empty_cache()

        # test data at 100, 200, 300
        if e%10 == 0 and e != 0:
            test_acc = []
            with torch.no_grad():
                for x_data, y_data in query_ds:
                    start_time = time.time()
                    x_data = x_data.to(device)
                    y_data = y_data.to(device)

                    # forwardpass
                    bb_rr  = backbone(x_data)
                    u_rr   = classifier_nk(bb_rr)

                    test_acc.append((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float())
                    del x_data, y_data, bb_rr, u_rr
                    torch.cuda.empty_cache()
            acc_all[int(e/10)-1] = torch.cat(test_acc).mean().item()*100
            print("|",end='')
    print("%d steps reached and the mean acc is %g , %g , %g"%(i, np.mean(np.array(acc_all[0])),np.mean(np.array(acc_all[1])),np.mean(np.array(acc_all[2])) ))


acc_mean1 = np.mean(acc_all[0])
acc_mean2 = np.mean(acc_all[1])
acc_mean3 = np.mean(acc_all[2])
acc_std1  = np.std(acc_all[0])
acc_std2  = np.std(acc_all[1])
acc_std3  = np.std(acc_all[2])
print('%d Test Acc at 100e= %4.2f%% +- %4.2f%%' %(args.iter_test, acc_mean1, 1.96* acc_std1/np.sqrt(args.iter_test)))
print('%d Test Acc at 200e= %4.2f%% +- %4.2f%%' %(args.iter_test, acc_mean2, 1.96* acc_std2/np.sqrt(args.iter_test)))
print('%d Test Acc at 300e= %4.2f%% +- %4.2f%%' %(args.iter_test, acc_mean3, 1.96* acc_std3/np.sqrt(args.iter_test)))

