import argparse, time, uuid

import torch
import numpy as np
from torchneuromorphic.torchneuromorphic.doublenmnist.doublenmnist_dataloaders import *
from torchneuromorphic.torchneuromorphic.dvs_asl.dvsasl_dataloaders import *
from training_curves import plot_curves

from lif_snn import backbone_conv_model, classifier_model, aux_task_gen

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    #torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")
dtype = torch.float32
ms = 1e-3

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--logfile", type=bool, default=False, help='Logfile on')
parser.add_argument("--batch-size", type=int, default=32, help='Batch size')
parser.add_argument("--epochs", type=int, default=401, help='Training Epochs') 
parser.add_argument("--burnin", type=int, default=10, help='Burnin Phase in ms')
parser.add_argument("--lr", type=float, default=1.0e-6, help='Learning Rate')
parser.add_argument("--lr-div", type=int, default=100, help='Learning Rate Division')
parser.add_argument("--log-int", type=int, default=10, help='Logging Interval')
parser.add_argument("--save-int", type=int, default=5, help='Checkpoint Save Interval')

# dataset
parser.add_argument("--dataset", type=str, default="ASL-DVS", help='Options: DNMNIST/ASL-DVS/DDVSGesture')
parser.add_argument("--train-samples", type=int, default=100, help='Number of samples per classes')
parser.add_argument("--n-train", type=int, default=14, help='N-way for training technically I guess more')
parser.add_argument("--downsampling", type=int, default=4, help='downsampling')

#architecture
parser.add_argument("--k1", type=int, default=5, help='Kernel Size 1')
parser.add_argument("--k2", type=int, default=5, help='Kernel Size 2')
parser.add_argument("--k3", type=int, default=5, help='Kernel Size 2')
parser.add_argument("--oc1", type=int, default=8, help='Output Channels 1')
parser.add_argument("--oc2", type=int, default=16, help='Output Channels 2')
parser.add_argument("--oc3", type=int, default=32, help='Output Channels 2')
parser.add_argument("--conv-bias", type=bool, default=True, help='Bias for conv layers')
parser.add_argument("--fc-bias", type=bool, default=True, help='Bias for classifier')
parser.add_argument("--init-gain-conv1", type=float, default=.5, help='Gain for weight init')
parser.add_argument("--init-gain-conv2", type=float, default=.5, help='Gain for weight init')
parser.add_argument("--init-gain-conv3", type=float, default=.5, help='Gain for weight init')
parser.add_argument("--init-gain-fc", type=float, default=1, help='Gain for weight init')

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


# training data
if args.dataset == 'DNMNIST':
    train_dl, test_dl  = sample_double_mnist_task(
                meta_dataset_type = 'train',
                N = args.n_train,
                K = args.train_samples,
                K_test = args.train_samples,
                root='data.nosync/nmnist/n_mnist.hdf5',
                batch_size=args.batch_size,
                batch_size_test=args.batch_size,
                ds=args.downsampling,
                num_workers=4)
elif args.dataset == 'ASL-DVS':
    train_dl, test_dl  = sample_dvsasl_task(
                meta_dataset_type = 'train',
                N = args.n_train,
                K = args.train_samples,
                K_test = args.train_samples,
                root='data.nosync/dvsasl/dvsasl.hdf5',
                batch_size=args.batch_size,
                batch_size_test=args.batch_size,
                ds=args.downsampling,
                num_workers=0)
elif args.dataset == 'DDVSGesture':
    train_dl, test_dl  = sample_double_mnist_task(
                meta_dataset_type = 'train',
                N = args.n_train,
                K = args.train_samples,
                K_test = args.train_samples,
                root='data.nosync/nmnist/n_mnist.hdf5',
                batch_size=args.batch_size,
                batch_size_test=args.batch_size,
                ds=args.downsampling,
                num_workers=4)
else:
    raise Exception("Invalid dataset")


x_preview, y_labels = next(iter(train_dl))


delta_t = args.delta_t*ms
T = x_preview.shape[1]
max_act = T - args.burnin
 
# backbone Conv
backbone = backbone_conv_model(x_preview = x_preview, in_channels = x_preview.shape[2], oc1 = args.oc1, oc2 = args.oc2, oc3 = args.oc3, k1 = args.k1, k2 = args.k2, k3 = args.k3, bias = args.conv_bias, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, thr = args.thr, reset = args.reset, gain1 = args.init_gain_conv1, gain2 = args.init_gain_conv2, gain3 = args.init_gain_conv3, delta_t = delta_t, dtype = dtype).to(device)

classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.n_train, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)

aux_classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = 4, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)



loss_fn = torch.nn.MSELoss(reduction = 'mean')
opt = torch.optim.Adam([
                {'params': backbone.parameters()},
                {'params': classifier.parameters()},
                {'params': aux_classifier.parameters()}
            ], lr = args.lr) #SGD before


acc_hist = []
aux_hist = []
clal_hist = []
auxl_hist = []


act1_hist = []
act2_hist = []
act3_hist = []
act4_hist = []

model_uuid = str(uuid.uuid4())

if args.logfile:
    with open("logs/train_"+model_uuid+".txt", "w+") as file_object:
        file_object.write(str(args) + "\n")
        file_object.write(model_uuid+ "\n")
        file_object.write("Start Training Backbone\n")
else:
    print(str(args))
    print(model_uuid)
    print("Start Training Backbone")

for e in range(args.epochs):
    e_time = time.time()
    avg_loss = avg_rloss = avg_s1 = avg_s2 = avg_s3 = avg_s4 = correct = rcorrect = total = 0

    # learning rate divide
    if e%args.lr_div == 0 and e != 0:
        for param_group in opt.param_groups:
            param_group['lr'] /= 2

    for i, (x_data, y_data) in enumerate(train_dl):
        start_time = time.time()
        x_data = x_data.to(device)

        # create aux task
        x_data, y_data, aux_y  = aux_task_gen(x_data, y_data)

        # forwardpass
        bb_rr  = backbone(x_data)
        u_rr   = classifier(bb_rr)
        aux_rr = aux_classifier(bb_rr)
        
        # class loss
        y_onehot = torch.zeros((u_rr.shape[0], u_rr.shape[2]), device = device).scatter_(1,  y_data.long().unsqueeze(dim = 1), (max_act*args.target_act) - (max_act*args.none_act)) + (max_act*args.none_act)
        #y_onehot = (y_data[:, ::100, :][:,0,:]* ((max_act*args.target_act) - (max_act*args.none_act))) + (max_act*args.none_act)
        class_loss = loss_fn(u_rr[:,args.burnin:,:].sum(dim = 1), y_onehot)

        # aux loss
        aux_y_onehot = torch.zeros((aux_rr.shape[0], aux_rr.shape[2]), device = device).scatter_(1,  aux_y.unsqueeze(dim = 1), (max_act*args.target_act) - (max_act*args.none_act)) + (max_act*args.none_act)
        aux_loss = loss_fn(aux_rr[:,args.burnin:,:].sum(dim = 1), aux_y_onehot)


        # BPTT
        loss = .5 * class_loss + .5 * aux_loss
        loss.backward()
        opt.step()
        opt.zero_grad()

        avg_loss = avg_loss + class_loss.data.item()
        avg_rloss = avg_rloss + aux_loss.data.item()
        avg_s1 = avg_s1 + np.sum(backbone.spike_count1[args.burnin:])/(T * backbone.f1_length)
        avg_s2 = avg_s2 + np.sum(backbone.spike_count2[args.burnin:])/(T * backbone.f2_length) 
        avg_s3 = avg_s3 + np.sum(backbone.spike_count3[args.burnin:])/(T * backbone.f_length) 
        avg_s4 = avg_s4 + np.sum(classifier.spike_count[args.burnin:])/(args.n_train*T)

        correct += (u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().sum()
        rcorrect += (aux_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().sum()
        total += x_data.shape[0]

        if i % args.log_int == 0:
            if args.logfile:
                with open("logs/train_"+model_uuid+".txt", "a") as file_object:
                    file_object.write('Epoch {:d} | Batch {:d}/{:d} | Loss {:.4f} | Rotate Loss {:.4f} | Accuracy {:4f} | Rotate Accuracy {:.4f} | Time {:.4f}\n'.format(e+1, i, len(train_dl), avg_loss/float(i+1), avg_rloss/float(i+1), (float(correct)*100)/total, (float(rcorrect)*100)/total, time.time() - start_time ))
            else:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:.4f} | Rotate Loss {:.4f} | Accuracy {:.4f} | Rotate Accuracy {:.4f} | Time {:.4f}'.format(e+1, i, len(train_dl), avg_loss/float(i+1), avg_rloss/float(i+1), (float(correct)*100)/total, (float(rcorrect)*100)/total, time.time() - start_time ))
        

    # accuracy on test
    with torch.no_grad():
        correct = rcorrect = total = 0
        for x_data, y_data in test_dl:

            start_time = time.time()
            x_data = x_data.to(device)

            # create aux task
            x_data, y_data, aux_y  = aux_task_gen(x_data, y_data)

            # forwardpass
            bb_rr  = backbone(x_data)
            u_rr   = classifier(bb_rr)
            aux_rr = aux_classifier(bb_rr)
            
            correct += (u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().sum()
            rcorrect += (aux_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().sum()
            total += x_data.shape[0]
    torch.cuda.empty_cache()

    # stats save
    acc_hist.append((float(correct)*100)/total)
    aux_hist.append((float(rcorrect)*100)/total)
    clal_hist.append(avg_loss/float(i+1))
    auxl_hist.append(avg_rloss/float(i+1))

    act1_hist.append(avg_s1/float(i+1))
    act2_hist.append(avg_s2/float(i+1))
    act3_hist.append(avg_s3/float(i+1))
    act4_hist.append(avg_s4/float(i+1))

    # logging and plotting
    if args.logfile:
        with open("logs/train_"+model_uuid+".txt", "a") as file_object:
            file_object.write("Epoch {:d} : Accuracy {:f}, Rotate Accuracy {:f}, Time {:f}\n".format(e+1,(float(correct)*100)/total,(float(rcorrect)*100)/total, time.time() - e_time))
    else:
        print("Epoch {:d} : Accuracy {:f}, Rotate Accuracy {:f}, Time {:f}".format(e+1,(float(correct)*100)/total,(float(rcorrect)*100)/total, time.time() - e_time))
    plot_curves(acc_hist, aux_hist, clal_hist, auxl_hist, act1_hist, act2_hist, act3_hist, act4_hist, model_uuid)

    # model save
    if e % args.save_int == 0:
        checkpoint_dict = {
                'backbone'     : backbone.state_dict(), 
                'classifer'    : classifier.state_dict(),
                'aux_class'    : aux_classifier.state_dict(),
                'optimizer'    : opt.state_dict(),
                'epoch'        : e, 
                'arguments'    : args,

                'loss_cla'     : clal_hist,
                'loss_aux'     : auxl_hist,
                'acc_cla'      : acc_hist,
                'acc_aux'      : aux_hist,

                's1c'          : act1_hist,
                's2c'          : act2_hist,
                's3c'          : act3_hist,
                's4c'          : act4_hist
        }
        torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
        del checkpoint_dict



# # test new data set
# from torchneuromorphic.dvs_asl.dvsasl_dataloaders import sample_dvsasl_task

# train_dl, test_dl  = sample_dvsasl_task(
#             meta_dataset_type = 'train',
#             N = 5,
#             K = 1,
#             K_test = 128,
#             root='../data.nosync/dvsasl/dvsasl.hdf5',
#             batch_size=72,
#             batch_size_test=72,
#             ds=1,
#             num_workers=0)

# x_preview, y_labels = next(iter(train_dl))


# # # # test new data set
# from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import create_dataloader

# train_dl, test_dl  = create_dataloader(
#     root = '../data.nosync/dvsgesture/dvs_gestures_build19.hdf5',
#     work_dir = '../data.nosync/dvsgesture/',
#     batch_size = 72 ,
#     chunk_size_train = 500,
#     chunk_size_test = 1800,
#     ds = 1,
#     dt = 1000,
#     num_workers=0)

# x_preview, y_labels = next(iter(train_dl))

# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# plt.clf()
# fig1 = plt.figure()

# ims = []
# for i in np.arange(x_preview.shape[1]):
#     #x_preview[0,i,0,:,:] *= -2
#     x_preview[0,i,1,:,:] += x_preview[0,i,0,:,:]

#     ims.append((plt.imshow(x_preview[0,i,1,:,:].cpu()),)) #t().flip(1)

# im_ani = animation.ArtistAnimation(fig1, ims, interval=50, repeat_delay=2000, blit=True)
# im_ani.save('/Users/clemens/Desktop/test_dvs.mp4')
