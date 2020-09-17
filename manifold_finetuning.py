import argparse, time, uuid
from tqdm import tqdm

import torch
import numpy as np
import torchneuromorphic.torchneuromorphic.doublenmnist.doublenmnist_dataloaders as dmnist
import torchneuromorphic.torchneuromorphic.dvs_asl.dvsasl_dataloaders as dvs_asl
import torchneuromorphic.torchneuromorphic.dvs_gestures.dvsgestures_dataloaders as dvs_gestures

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
parser.add_argument("--self-supervision", type=bool, default=True, help='Logfile on')
parser.add_argument("--batch-size", type=int, default=138, help='Batch size')
parser.add_argument("--epochs", type=int, default=401, help='Training Epochs') 
parser.add_argument("--alpha", type=int, default=2, help='Alpha for manifold mixup') 
parser.add_argument("--burnin", type=int, default=30, help='Burnin Phase in ms')
parser.add_argument("--lr", type=float, default=1.0e-12, help='Learning Rate')
parser.add_argument("--lr-div", type=int, default=20, help='Learning Rate Division')
parser.add_argument("--log-int", type=int, default=5, help='Logging Interval')
parser.add_argument("--save-int", type=int, default=5, help='Checkpoint Save Interval')
parser.add_argument("--train-tau", type=bool, default=False, help='Train time constants')

# dataset
parser.add_argument("--dataset", type=str, default="DNMNIST", help='Options: DNMNIST/ASL-DVS/DDVSGesture')
parser.add_argument("--train-samples", type=int, default=125, help='Number of samples per classes')
parser.add_argument("--n-train", type=int, default=64, help='N-way for training technically I guess more')
parser.add_argument("--downsampling", type=int, default=2, help='downsampling')

#architecture
parser.add_argument("--k1", type=int, default=7, help='Kernel Size 1')
parser.add_argument("--k2", type=int, default=7, help='Kernel Size 2')
parser.add_argument("--k3", type=int, default=7, help='Kernel Size 2')
parser.add_argument("--oc1", type=int, default=32, help='Output Channels 1')
parser.add_argument("--oc2", type=int, default=64, help='Output Channels 2')
parser.add_argument("--oc3", type=int, default=64, help='Output Channels 2')
parser.add_argument("--padding", type=int, default=2, help='Conv Padding')
parser.add_argument("--conv-bias", type=bool, default=True, help='Bias for conv layers')
parser.add_argument("--fc-bias", type=bool, default=True, help='Bias for classifier')
parser.add_argument("--init-gain-conv1", type=float, default=1e-10, help='Gain for weight init 1 conv')
parser.add_argument("--init-gain-conv2", type=float, default=1e-10, help='Gain for weight init 2 conv')
parser.add_argument("--init-gain-conv3", type=float, default=1e-10, help='Gain for weight init 3 conv')
parser.add_argument("--init-gain-fc", type=float, default=1e-10, help='Gain for weight init fc')
parser.add_argument("--init-gain-aux", type=float, default=1e-10, help='Gain for weight init fc')

# neural dynamics
parser.add_argument("--delta-t", type=int, default=1, help='Time steps')
parser.add_argument("--tau-mem-low", type=float, default=35, help='Membrane time constant low')
parser.add_argument("--tau-syn-low", type=float, default=10, help='Synaptic time constant low')
parser.add_argument("--tau-ref-low", type=float, default=2.5, help='Refractory time constant low')
parser.add_argument("--tau-mem-high", type=float, default=35, help='Membrane time constant high')
parser.add_argument("--tau-syn-high", type=float, default=10, help='Synaptic time constant high')
parser.add_argument("--tau-ref-high", type=float, default=2.5, help='Refractory time constant high')
parser.add_argument("--thr", type=float, default=.0, help='Firing Threshold')

args = parser.parse_args()

args.tau_mem_low = args.tau_mem_high
args.tau_syn_low = args.tau_syn_high
args.tau_ref_low = args.tau_ref_high

args.init_gain_conv2 = args.init_gain_conv3 = args.init_gain_fc = args.init_gain_aux = args.init_gain_conv1  


# training data
if args.dataset == 'DNMNIST':
    train_dl, test_dl  = dmnist.sample_double_mnist_task(
                meta_dataset_type = 'train',
                N = args.n_train,
                K = args.train_samples,
                K_test = args.train_samples,
                root='data.nosync/nmnist/n_mnist.hdf5',
                batch_size=args.batch_size,
                batch_size_test=args.batch_size,
                ds=args.downsampling,
                num_workers=4)
    time_steps_train = 300
    time_steps_test = 300
    one_hot_opt = True
elif args.dataset == 'ASL-DVS':
    train_dl, test_dl  = dvsasl_dataloaders.sample_dvsasl_task(
                meta_dataset_type = 'train',
                N = args.n_train,
                K = args.train_samples,
                K_test = args.train_samples,
                root='data.nosync/dvsasl/dvsasl.hdf5',
                batch_size=args.batch_size,
                batch_size_test=args.batch_size,
                ds=args.downsampling,
                num_workers=4)
    time_steps_train = 100
    time_steps_test = 100
    one_hot_opt = True
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
    time_steps_train = 500
    time_steps_test = 1800
    one_hot_opt = True
elif args.dataset == 'DVSGesture':
    train_dl, test_dl  = dvs_gestures.create_dataloader(
                root = 'data.nosync/dvsgesture/dvs_gestures_build.hdf5',
                work_dir = 'data/dvsgesture/',
                batch_size = args.batch_size,
                chunk_size_train = 500,
                chunk_size_test = 1800,
                ds = args.downsampling,
                dt = 1000,
                transform_train = None,
                transform_test = None,
                target_transform_train = None,
                target_transform_test = None,
                n_events_attention=None,
                num_workers=4)
    time_steps_train = 500
    time_steps_test = 1800
    one_hot_opt = False
else:
    raise Exception("Invalid dataset")

x_preview, y_labels = next(iter(train_dl))
model_uuid = str(uuid.uuid4())   

delta_t = args.delta_t*ms
T = x_preview.shape[1]

backbone = backbone_conv_model(x_preview = x_preview, in_channels = x_preview.shape[2], oc1 = args.oc1, oc2 = args.oc2, oc3 = args.oc3, k1 = args.k1, k2 = args.k2, k3 = args.k3, padding = args.padding, bias = args.conv_bias, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, thr = args.thr, gain1 = args.init_gain_conv1, gain2 = args.init_gain_conv2, gain3 = args.init_gain_conv3, delta_t = delta_t, train_t = args.train_tau, dtype = dtype).to(device)

classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.n_train, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, train_t = args.train_tau, dtype = dtype).to(device)

aux_classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = 4, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, thr = args.thr, gain = args.init_gain_aux, delta_t = delta_t, train_t = args.train_tau, dtype = dtype).to(device)


loss_fn = torch.nn.NLLLoss()
softmax_pass = torch.nn.LogSoftmax(dim=1)
opt = torch.optim.Adam([
                {'params': backbone.parameters()},
                {'params': classifier.parameters()},
                {'params': aux_classifier.parameters()}
            ], lr = args.lr)


acc_hist = []
mm_acc_hist = []
aux_hist = []
clal_hist = []
auxl_hist = []

act1_hist = []
act2_hist = []
act3_hist = []
act4_hist = []
actA_hist = []

if args.logfile:
    with open("logs/train_"+model_uuid+".txt", "w+") as file_object:
        file_object.write(str(args) + "\n")
        file_object.write(str(args) + "\n")
        file_object.write("Start Manifold Training Backbone\n")
        file_object.write(model_uuid+ "\n")
else:
    print(str(args))
    print(str(args))

    print("Start Manifold Backbone Training Backbone")
    print(model_uuid)

for e in range(args.epochs):
    e_time = time.time()
    avg_loss = avg_rloss = avg_s1 = avg_s2 = avg_s3 = avg_s4 = avg_A = correct = mm_correct = rcorrect = total = 0

    # learning rate divide
    if e%args.lr_div == 0 and e != 0:
        for param_group in opt.param_groups:
            param_group['lr'] /= 2
            args.lr_div *= 2

    for i, (x_data, y_data) in enumerate(train_dl):
        start_time = time.time()
        opt.zero_grad()
        x_data = x_data.to(device)

        # manifold mixup
        lam = np.random.beta(args.alpha, args.alpha)
        layer_mix = np.random.randint(0,3)
        index = torch.randperm(x_data.shape[0]).to(device)

        y_a, y_b = y_data.to(device), y_data[index].to(device)
        
        # forwardpass
        backbone.state_init_net(x_data.shape[0], device)
        s_t = torch.zeros((x_data.shape[0], backbone.T, backbone.f_length), device = device)
        for t in range(backbone.T):
            x = x_data[:,t,:,:,:]
            if layer_mix == 0:
                x = lam * x + (1 - lam) * x[index,:]
            x, _       = backbone.conv_layer1.forward(x)
            x          = backbone.mpooling(x)
            backbone.spike_count1[t] += int(x.view(x.shape[0], -1).sum(dim=1).mean().item())
            if layer_mix == 1:
                x = lam * x + (1 - lam) * x[index,:]
            x, _       = backbone.conv_layer2.forward(x)
            backbone.spike_count2[t] += int(x.view(x.shape[0], -1).sum(dim=1).mean().item())
            if layer_mix == 2:
                x = lam * x + (1 - lam) * x[index,:]
            x, _       = backbone.conv_layer3.forward(x)
            x          = backbone.mpooling(x)
            backbone.spike_count3[t] += int(x.view(x.shape[0], -1).sum(dim=1).mean().item())
            s_t[:,t,:] = x.view(-1, backbone.f_length)

        u_rr   = classifier(s_t)

        mm_loss = lam * loss_fn( softmax_pass(u_rr[:,args.burnin:,:].sum(dim = 1)), y_a) + (1 - lam) * loss_fn( softmax_pass(u_rr[:,args.burnin:,:].sum(dim = 1)), y_b)
        mm_loss.backward()

        mm_correct += lam * float((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_a).float().sum()) + (1 - lam) * float((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_b).float().sum())

        avg_loss = avg_loss + float(mm_loss.item()) 

        del s_t, y_a, y_b, u_rr
        torch.cuda.empty_cache()

        # Rotation Training
        if args.self_supervision:
            # create aux task
            x_data, y_data, aux_y  = aux_task_gen(x_data, y_data)
            aux_y = aux_y.to(device)

        # to gpu
        y_data = y_data.to(device)
        
        # forward pass
        bb_rr  = backbone(x_data)
        del x_data
        torch.cuda.empty_cache()

        if args.self_supervision:
            aux_rr = aux_classifier(bb_rr)
            # aux loss
            aux_loss = loss_fn( softmax_pass(aux_rr[:,args.burnin:,:].sum(dim = 1)), aux_y)
            rcorrect += float((aux_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().sum())

        u_rr   = classifier(bb_rr)
        class_loss = loss_fn(softmax_pass(u_rr[:,args.burnin:,:].sum(dim = 1)), y_data)

        # BPTT
        if args.self_supervision:
            loss = (class_loss + aux_loss)/2
        else:
            loss = class_loss
        loss.backward()
        opt.step()

        correct += float((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().sum())
        total += float(y_data.shape[0])

        # stats
        avg_loss = avg_loss + float(loss.data.item()) 
        avg_s1 = avg_s1 + float(np.sum(backbone.spike_count1[args.burnin:])/(T * backbone.f1_length))
        avg_s2 = avg_s2 + float(np.sum(backbone.spike_count2[args.burnin:])/(T * backbone.f2_length))
        avg_s3 = avg_s3 + float(np.sum(backbone.spike_count3[args.burnin:])/(T * backbone.f_length)) 
        avg_s4 = avg_s4 + float(np.sum(classifier.spike_count[args.burnin:])/(args.n_train*T))
        if args.self_supervision:
            avg_A = avg_A + float(np.sum(aux_classifier.spike_count[args.burnin:])/(args.n_train*T))
            avg_rloss = avg_rloss + float(aux_loss.data.item())
        
        del aux_rr, aux_y, bb_rr, u_rr, mm_loss, loss, class_loss, aux_loss, y_data
        torch.cuda.empty_cache()

        if i % args.log_int == 0:
            if args.logfile:
                with open("logs/train_"+model_uuid+".txt", "a") as file_object:
                    file_object.write('Epoch {:d} | Batch {:d}/{:d} | Loss {:.4f} | Rotate Loss {:.4f} | Accuracy {:4f}/{:4f} | Rotate Accuracy {:.4f} | Time {:.4f}\n'.format(e+1, i, len(train_dl), avg_loss/float(i+1), avg_rloss/float(i+1), (float(correct)*100)/total, (float(mm_correct)*100)/total, (float(rcorrect)*100)/total, time.time() - start_time ))
            else:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:.4f} | Rotate Loss {:.4f} | Accuracy {:.4f}/{:.4} | Rotate Accuracy {:.4f} | Time {:.4f}'.format(e+1, i, len(train_dl), avg_loss/float(i+1), avg_rloss/float(i+1), (float(correct)*100)/total, (float(mm_correct)*100)/total, (float(rcorrect)*100)/total, time.time() - start_time ))
        

    # accuracy on test
    with torch.no_grad():
        correct = rcorrect = total = 0
        for x_data, y_data in test_dl:

            start_time = time.time()
            x_data = x_data#.to(device)

            if args.self_supervision:
                # create aux task
                x_data, y_data, aux_y  = aux_task_gen(x_data, y_data)
                aux_y = aux_y.to(device)

            # to gpu
            x_data = x_data.to(device)
            y_data = y_data.to(device)

            # forwardpass
            bb_rr  = backbone(x_data)
            del x_data
            u_rr   = classifier(bb_rr)
            if args.self_supervision:
                aux_rr = aux_classifier(bb_rr)
                rcorrect += float((aux_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().sum())
            
            correct += float((u_rr[:,args.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().sum())
            total += float(y_data.shape[0])
    torch.cuda.empty_cache()

    # stats save
    mm_acc_hist.append((float(mm_correct)*100)/total)
    acc_hist.append((float(correct)*100)/total)
    aux_hist.append((float(rcorrect)*100)/total)
    clal_hist.append(avg_loss/float(i+1))
    auxl_hist.append(avg_rloss/float(i+1))

    act1_hist.append(avg_s1/float(i+1))
    act2_hist.append(avg_s2/float(i+1))
    act3_hist.append(avg_s3/float(i+1))
    act4_hist.append(avg_s4/float(i+1))
    actA_hist.append(avg_A/float(i+1))

    # logging and plotting
    if args.logfile:
        with open("logs/train_"+model_uuid+".txt", "a") as file_object:
            file_object.write("Epoch {:d} : Accuracy {:f}, Rotate Accuracy {:f}, Time {:f}\n".format(e+1,(float(correct)*100)/total,(float(rcorrect)*100)/total, time.time() - e_time))
    else:
        print("Epoch {:d} : Accuracy {:f}, Rotate Accuracy {:f}, Time {:f}".format(e+1,(float(correct)*100)/total,(float(rcorrect)*100)/total, time.time() - e_time))
        print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(act1_hist[-1], act2_hist[-1], act3_hist[-1], act4_hist[-1], actA_hist[-1]))
    plot_curves(acc_hist, aux_hist, clal_hist, auxl_hist, act1_hist, act2_hist, act3_hist, act4_hist, actA_hist, model_uuid)


    # sace model which does best on the few shot learning

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
                'mm_acc'       : mm_acc_hist,

                's1c'          : act1_hist,
                's2c'          : act2_hist,
                's3c'          : act3_hist,
                's4c'          : act4_hist
        }
        torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
        del checkpoint_dict