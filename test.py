import argparse, time, uuid

import torch
import numpy as np
from torchneuromorphic.torchneuromorphic.doublenmnist.doublenmnist_dataloaders import *
from tqdm import tqdm

from lif_snn import backbone_conv_model, classifier_model

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    #torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")
dtype = torch.float32
ms = 1e-3

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--checkpoint", type=str, default='2b29b60b-100c-40b2-aa9d-ec083fa3a2e9', help='UUID for checkpoint to be tested')

parser.add_argument("--epochs", type=int, default=91, help='Training Epochs') 
parser.add_argument("--batch-size-test", type=int, default=4, help='Batch size')
parser.add_argument("--batch-size-test-test", type=int, default=128, help='Batch size test test')
parser.add_argument("--progressbar-off", type=float, default=False, help='False: progressbar activated')

parser.add_argument("--iter-test", type=int, default=300, help='Test Iter')
parser.add_argument("--burnin", type=int, default=10, help='Burnin Phase in ms')
parser.add_argument("--lr", type=float, default=1.0e-7, help='Learning Rate')
parser.add_argument("--init-gain-fc", type=float, default=1, help='Gain for weight init')
parser.add_argument("--init-gain-backbone", type=float, default=.5, help='Gain for weight init')

parser.add_argument("--test-samples", type=int, default=100, help='Number of samples per classes')
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
parser.add_argument("--reset", type=float, default=1, help='Reset strength')
parser.add_argument("--thr", type=float, default=.5, help='Firing Threshold')
parser.add_argument("--target_act", type=float, default=.95, help='Firing Threshold')
parser.add_argument("--none_act", type=float, default=.05, help='Firing Threshold')

args = parser.parse_args()
model_uuid = args.checkpoint

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

backbone = backbone_conv_model(x_preview = x_preview, in_channels = x_preview.shape[2], oc1 = args.oc1, oc2 = args.oc2, k1 = args.k1, k2 = args.k2, bias = args.conv_bias, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, thr = args.thr, reset = args.reset, gain = args.init_gain_backbone, delta_t = delta_t, dtype = dtype).to(device)

# load backbone
checkpoint_dict = torch.load('./checkpoints/'+ args.checkpoint +'.pkl')
backbone.load_state_dict(checkpoint_dict['backbone'])
e = checkpoint_dict['epoch']
del checkpoint_dict

with open("logs/test_"+model_uuid+".txt", "a") as file_object:
    file_object.write("\nEvaluating model %s at epoch %d over %d classes with %d examples\n"%(args.checkpoint, e, args.n_way, args.k_shot))

acc_all = [[],[],[]]
for i in range(args.iter_test):
    start_time = time.time()

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

    # new classifier
    classifier_nk = classifier_model(T = T, inp_neurons = backbone.f_length, output_classes = args.n_way, tau_ref_low = args.tau_ref_low*ms, tau_mem_low = args.tau_mem_low*ms, tau_syn_low = args.tau_syn_low*ms, tau_ref_high = args.tau_ref_high*ms, tau_mem_high = args.tau_mem_high*ms, tau_syn_high = args.tau_syn_high*ms, bias = args.fc_bias, reset = args.reset, thr = args.thr, gain = args.init_gain_fc, delta_t = delta_t, dtype = dtype).to(device)

    loss_fn = torch.nn.MSELoss(reduction = 'mean')
    opt = torch.optim.Adam(classifier_nk.parameters(), lr = args.lr)

    for e in tqdm(range(args.epochs), disable = args.progressbar_off):
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
        if e%30 == 0 and e != 0:
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
    with open("logs/test_"+model_uuid+".txt", "a") as file_object:
        file_object.write("%d steps reached and the mean acc is %g , %g , %g time %4.2f%%\n"%(i, np.mean(np.array(acc_all[0])),np.mean(np.array(acc_all[1])),np.mean(np.array(acc_all[2])), time.time() - start_time))


acc_mean1 = np.mean(acc_all[0])
acc_mean2 = np.mean(acc_all[1])
acc_mean3 = np.mean(acc_all[2])
acc_std1  = np.std(acc_all[0])
acc_std2  = np.std(acc_all[1])
acc_std3  = np.std(acc_all[2])
with open("logs/test_"+model_uuid+".txt", "a") as file_object:
    file_object.write('%d Test Acc at 100e= %4.2f%% +- %4.2f%%\n' %(args.iter_test, acc_mean1, 1.96* acc_std1/np.sqrt(args.iter_test)))
    file_object.write('%d Test Acc at 200e= %4.2f%% +- %4.2f%%\n' %(args.iter_test, acc_mean2, 1.96* acc_std2/np.sqrt(args.iter_test)))
    file_object.write('%d Test Acc at 300e= %4.2f%% +- %4.2f%%\n' %(args.iter_test, acc_mean3, 1.96* acc_std3/np.sqrt(args.iter_test)))

