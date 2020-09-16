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
parser.add_argument("--checkpoint", type=str, default='0c41322f-6e60-4005-bd53-1a3396e74de5', help='UUID for checkpoint to be tested')
parser.add_argument("--from-scratch", type=bool, default=True, help='Start fraining from scratch')
parser.add_argument("--lr", type=float, default=1.0e-12, help='Learning Rate')
parser.add_argument("--lr-div", type=int, default=100, help='Learning Rate Division')
parser.add_argument("--epochs", type=int, default=151, help='Training Epochs') 
parser.add_argument("--alpha", type=int, default=2, help='Training Epochs') 
parser.add_argument("--log-int", type=int, default=5, help='Logging Interval')
parser.add_argument("--save-int", type=int, default=5, help='Checkpoint Save Interval')
args = parser.parse_args()

checkpoint_dict = torch.load('./checkpoints/'+ args.checkpoint +'.pkl')
args_loaded = checkpoint_dict['arguments']

# training data
if args_loaded.dataset == 'DNMNIST':
    train_dl, test_dl  = dmnist.sample_double_mnist_task(
                meta_dataset_type = 'train',
                N = args_loaded.n_train,
                K = args_loaded.train_samples,
                K_test = args_loaded.train_samples,
                root='data.nosync/nmnist/n_mnist.hdf5',
                batch_size=args_loaded.batch_size,
                batch_size_test=args_loaded.batch_size,
                ds=args_loaded.downsampling,
                num_workers=4)
    time_steps_train = 300
    time_steps_test = 300
    one_hot_opt = True
elif args_loaded.dataset == 'ASL-DVS':
    train_dl, test_dl  = dvsasl_dataloaders.sample_dvsasl_task(
                meta_dataset_type = 'train',
                N = args_loaded.n_train,
                K = args_loaded.train_samples,
                K_test = args_loaded.train_samples,
                root='data.nosync/dvsasl/dvsasl.hdf5',
                batch_size=args_loaded.batch_size,
                batch_size_test=args_loaded.batch_size,
                ds=args_loaded.downsampling,
                num_workers=4)
    time_steps_train = 100
    time_steps_test = 100
    one_hot_opt = True
elif args_loaded.dataset == 'DDVSGesture':
    train_dl, test_dl  = sample_double_mnist_task(
                meta_dataset_type = 'train',
                N = args_loaded.n_train,
                K = args_loaded.train_samples,
                K_test = args_loaded.train_samples,
                root='data.nosync/nmnist/n_mnist.hdf5',
                batch_size=args_loaded.batch_size,
                batch_size_test=args_loaded.batch_size,
                ds=args_loaded.downsampling,
                num_workers=4)
    time_steps_train = 500
    time_steps_test = 1800
    one_hot_opt = True
elif args_loaded.dataset == 'DVSGesture':
    train_dl, test_dl  = dvs_gestures.create_dataloader(
                root = 'data.nosync/dvsgesture/dvs_gestures_build.hdf5',
                work_dir = 'data/dvsgesture/',
                batch_size = args_loaded.batch_size,
                chunk_size_train = 500,
                chunk_size_test = 1800,
                ds = args_loaded.downsampling,
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

delta_t = args_loaded.delta_t*ms
T = x_preview.shape[1]

backbone = backbone_conv_model(x_preview = x_preview, in_channels = x_preview.shape[2], oc1 = args_loaded.oc1, oc2 = args_loaded.oc2, oc3 = args_loaded.oc3, k1 = args_loaded.k1, k2 = args_loaded.k2, k3 = args_loaded.k3, padding = args_loaded.padding, bias = args_loaded.conv_bias, tau_ref_low = args_loaded.tau_ref_low*ms, tau_mem_low = args_loaded.tau_mem_low*ms, tau_syn_low = args_loaded.tau_syn_low*ms, tau_ref_high = args_loaded.tau_ref_high*ms, tau_mem_high = args_loaded.tau_mem_high*ms, tau_syn_high = args_loaded.tau_syn_high*ms, thr = args_loaded.thr, gain1 = args_loaded.init_gain_conv1, gain2 = args_loaded.init_gain_conv2, gain3 = args_loaded.init_gain_conv3, delta_t = delta_t, train_t = args_loaded.train_tau, dtype = dtype).to(device)

classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args_loaded.n_train, tau_ref_low = args_loaded.tau_ref_low*ms, tau_mem_low = args_loaded.tau_mem_low*ms, tau_syn_low = args_loaded.tau_syn_low*ms, tau_ref_high = args_loaded.tau_ref_high*ms, tau_mem_high = args_loaded.tau_mem_high*ms, tau_syn_high = args_loaded.tau_syn_high*ms, bias = args_loaded.fc_bias, thr = args_loaded.thr, gain = args_loaded.init_gain_fc, delta_t = delta_t, train_t = args_loaded.train_tau, dtype = dtype).to(device)

aux_classifier = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = 4, tau_ref_low = args_loaded.tau_ref_low*ms, tau_mem_low = args_loaded.tau_mem_low*ms, tau_syn_low = args_loaded.tau_syn_low*ms, tau_ref_high = args_loaded.tau_ref_high*ms, tau_mem_high = args_loaded.tau_mem_high*ms, tau_syn_high = args_loaded.tau_syn_high*ms, bias = args_loaded.fc_bias, thr = args_loaded.thr, gain = args_loaded.init_gain_aux, delta_t = delta_t, train_t = args_loaded.train_tau, dtype = dtype).to(device)

# load backbone
if not args.from_scratch:
    backbone.load_state_dict(checkpoint_dict['backbone'])
    classifier.load_state_dict(checkpoint_dict['classifer'])
    aux_classifier.load_state_dict(checkpoint_dict['aux_class'])
del checkpoint_dict


loss_fn = torch.nn.NLLLoss()
softmax_pass = torch.nn.LogSoftmax(dim=1)
opt = torch.optim.Adam([
                {'params': backbone.parameters()},
                {'params': classifier.parameters()},
                {'params': aux_classifier.parameters()}
            ], lr = args.lr)


acc_hist = []
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
        file_object.write(str(args_loaded) + "\n")
        file_object.write("Training based on "+ args.checkpoint + "\n")
        file_object.write("Start Manifold Training Backbone\n")
        file_object.write(model_uuid+ "\n")
else:
    print(str(args))
    print(str(args_loaded))
    print("Training based on "+ args.checkpoint)
    print("Start Manifold Backbone Training Backbone")
    print(model_uuid)

for e in range(args.epochs):
    e_time = time.time()
    avg_loss = avg_rloss = avg_s1 = avg_s2 = avg_s3 = avg_s4 = avg_A = correct = rcorrect = total = 0

    # learning rate divide
    if e%args.lr_div == 0 and e != 0:
        for param_group in opt.param_groups:
            param_group['lr'] /= 2

    for i, (x_data, y_data) in enumerate(train_dl):
        start_time = time.time()

        # manifold mixup
        lam = np.random.beta(args.alpha, args.alpha)
        layer_mix = np.random.randint(0,3)
        index = torch.randperm(x_data.shape[0]).to(device)

        y_a, y_b = y_data.to(device), y_data[index].to(device)
        
        # forwardpass
        backbone.state_init_net()
        s_t = torch.zeros((inputs.shape[0], backbone.T, backbone.f_length), device = inputs.device)
        for t in range(backbone.T):
            x = x_data[:,t,:,:,:].to(device)
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

        mm_loss = lam * loss_fn( softmax_pass(u_rr[:,args_loaded.burnin:,:].sum(dim = 1)), y_a) + (1 - lam) * loss_fn( softmax_pass(u_rr[:,args_loaded.burnin:,:].sum(dim = 1)), y_b)
        #mm_correct = 

        # Rotation Training
        if args_loaded.self_supervision:
            # create aux task
            x_data, y_data, aux_y  = aux_task_gen(x_data, y_data)
            aux_y = aux_y.to(device)

        # to gpu
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        
        # forwardpass
        bb_rr  = backbone(x_data)
        del x_data
        torch.cuda.empty_cache()
        u_rr   = classifier(bb_rr)

        # class loss
        class_loss = loss_fn( softmax_pass(u_rr[:,args.burnin:,:].sum(dim = 1)), y_data)

        if args_loaded.self_supervision:
            aux_rr = aux_classifier(s_t)

            # aux loss
            aux_loss = loss_fn( softmax_pass(aux_rr[:,args_loaded.burnin:,:].sum(dim = 1)), aux_y)
            rcorrect += float((aux_rr[:,args_loaded.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().sum())
        # BPTT
        if args_loaded.self_supervision:
            loss = mm_loss + .5 * (class_loss + aux_loss)
        else:
            loss = mm_loss + class_loss
        loss.backward()
        opt.step()
        opt.zero_grad()

        del s_t, y_a, y_b, aux_rr, aux_y
        torch.cuda.empty_cache()

        correct += lam * float((u_rr[:,args_loaded.burnin:,:].sum(dim = 1).argmax(dim=1) == y_a).float().sum()) + (1 - lam) * float((u_rr[:,args_loaded.burnin:,:].sum(dim = 1).argmax(dim=1) == y_b).float().sum())
        total += float(y_data.shape[0])

        # update taus
        if args_loaded.train_tau:
            backbone.update_taus() 
            classifier.update_taus()
            aux_classifier.update_taus()

        # stats
        avg_loss = avg_loss + float(loss.data.item())
        avg_s1 = avg_s1 + float(np.sum(backbone.spike_count1[args_loaded.burnin:])/(T * backbone.f1_length))
        avg_s2 = avg_s2 + float(np.sum(backbone.spike_count2[args_loaded.burnin:])/(T * backbone.f2_length))
        avg_s3 = avg_s3 + float(np.sum(backbone.spike_count3[args_loaded.burnin:])/(T * backbone.f_length)) 
        avg_s4 = avg_s4 + float(np.sum(classifier.spike_count[args_loaded.burnin:])/(args_loaded.n_train*T))
        if args_loaded.self_supervision:
            avg_A = avg_A + float(np.sum(aux_classifier.spike_count[args_loaded.burnin:])/(args_loaded.n_train*T))
            avg_rloss = avg_rloss + float(aux_loss.data.item())
        
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
            x_data = x_data#.to(device)

            if args_loaded.self_supervision:
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
            if args_loaded.self_supervision:
                aux_rr = aux_classifier(bb_rr)
                rcorrect += float((aux_rr[:,args_loaded.burnin:,:].sum(dim = 1).argmax(dim=1) == aux_y).float().sum())
            
            correct += float((u_rr[:,args_loaded.burnin:,:].sum(dim = 1).argmax(dim=1) == y_data).float().sum())
            total += float(y_data.shape[0])
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

                's1c'          : act1_hist,
                's2c'          : act2_hist,
                's3c'          : act3_hist,
                's4c'          : act4_hist
        }
        torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
        del checkpoint_dict