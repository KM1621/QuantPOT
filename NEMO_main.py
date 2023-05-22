#!/usr/bin/env python
# coding: utf-8

# This example guides a first-time user of NEMO into the NEMO quantization process, using a small pretrained network and going through post-training per-layer quantization (i.e., representing weight and activation tensors as integers) and deployment (i.e., organizing operations so that they are an accurate representation of behavior in integer-based hardware). We will see how this operates through four stages: *FullPrecision*, *FakeQuantized*, *QuantizedDeployable*, *IntegerDeployable*.
# 
# NEMO uses `float32` tensors to represent data at all four stages - including *IntegerDeployable*. This means that NEMO code does not need special hardware support for integers to run on GPUs. It also means that NEMO is not (and does not want to be) a runtime for quantized neural networks on embedded systems!
# 
# Let us start by installing dependencies...

# pip install -e .
import argparse
import time
import shutil
import os
import matplotlib.pyplot as plt
# ... and import all packages, including NEMO itself:
from nemo.quant.pact import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import nemo
from tqdm import tqdm

import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch MNIST Nemo')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('--qat', default=False, type=str, help='True for QAT and False for full precision training')
parser.add_argument('--ptq', default=True, type=str, help='True for PTQ and False for full precision inference')


# The first real step is to define the network topology. This works exactly like in a "standard" PyTorch script, using regular `torch.nn.Module` instances.  NEMO can transform many layers defined in `torch.nn` into its own representations. There are a few constraints, however, to the network construction:
# * Use `torch.nn.Module`, not `torch.autograd.Function`: NEMO works by listing modules and ignores functions by construction. Everything coming from the `torch.nn.functional` library is ignored by NEMO: for example, you have to use `torch.nn.ReLU` module instead of the equivalent `torch.nn.functional.relu` function, which is often found in examples online.
# * Instantiate a separate `torch.nn.Module` for each node in your topology; you already do this for parametric modules (e.g., `torch.nn.Conv2d`), but you have to do the same also for `torch.nn.ReLU` and other parameterless modules. NEMO will introduce quantization parameters that will change along the network.
# * To converge two network branches (e.g., a main and a residual branch), a normal PyTorch network will usually add the values of their output tensors. This will keep working for a network at the *FakeQuantized* stage, i.e., one that can be fine-tuned keeping into account quantization - but it will break in later stages. In the *QuantizedDeployable* and *IntegerDeployable* stages, the branch reconvergence has to take into account the possibly different precision of the two branches, therefore NEMO has to know that there is an "Add" node at that point of the network. This can be realized using the `nemo.quant.pact.PACT_IntegerAdd` module, which is entirely equivalent to a normal addition in *FullPrecision* and *FakeQuantized* stages. 
 
# 
best_prec = 0
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./output', exist_ok=True)

def main():

    global args, best_prec
    
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    # Download a pretrained model, and test it! Here we operate at what we call the ***FullPrecision*** stage: the regular PyTorch representation, which relies on real-valued tensors represented by `float32` in your CPU/GPU.
    if args.bit == 2:
        
        data = {
                'Experiment': ['Full precision pretrained', 'FakeQuantized (First Try)', 'FakeQuantized (Final epoch)', 'FakeQuantized (calibrated)', 'FakeQuantized (folded)', 'QuantizedDeployable', 'QuantizedDeployable (calibrated)', 'IntegerDeployable', 'IntegerDeployable (Real)'],
                'acc_' + str(args.bit): [0., 0., 0., 0., 0., 0., 0., 0., 0.]
                }
        row_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        df = pd.DataFrame(data=data, index=row_labels)
    else:
        
        data = pd.read_csv('./output/accuracy_tracesQAT' + str(args.qat) + '.csv')
        data['acc_' + str(args.bit)] = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        df = pd.DataFrame(data=data)

    print(df)
    
    model = ExampleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_qerr = nn.MSELoss().cuda()
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs
    )

    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            state_dict = torch.load(args.init, map_location=device)
            print("=> testing pre-trained model")
            #state_dict = torch.load("mnist_cnn_fp.pt", map_location=device)
            model.load_state_dict(state_dict, strict=True)
            acc = test(model, device, test_loader)
            df.loc[0,'acc_' + str(args.bit)] = acc
            print("\nFullPrecision accuracy of a pretrained model: %.02f%%" % acc)
            
        else:
            print('No pre-trained model found !')
            exit()

    if args.evaluate:
        test(model, device, test_loader, verbose=True)
        acc = test(model, device, test_loader)
        print("\nAccuracy from evaluation only mode: %.02f%%" % acc)
        #print("\nFakeQuantized @ 16b accuracy (first try): %.02f%%" % acc)
        return
    # The first try looks... not so good. 82% is actually pretty bad for MNIST! What happened? Remember that while clipping parameters for weights can be set statically, this is not true for activations: so the missing piece is the characterization of activation clipping ($\alpha$ parameter), which is currently set to a default value.
    # 
    # In NEMO, this initial calibration can be performed by setting a special *statistics collection* mode for activations, which is used to explicitly reset the $\alpha$ params. The calibration is performed directly by running inference over a dataset; in this case, we cheat a bit and do it on the test set.


    if args.ptq:
        model = nemo.transform.quantize_pact(model, dummy_input=torch.randn((1,1,28,28)).to(device))
        precision = {
            'conv1': {
                'W_bits' : args.bit
            },
            'conv2': {
                'W_bits' : args.bit
            },
            'fc1': {
                'W_bits' : args.bit
            },
            'fc2': {
                'W_bits' : args.bit
            },
            'relu1': {
                'x_bits' : args.bit
            },
            'relu2': {
                'x_bits' : args.bit
            },
            'fcrelu1': {
                'x_bits' : args.bit
            },
        }
        model.change_precision(bits=1, min_prec_dict=precision)
        acc = test(model, device, test_loader)
        print("\nFakeQuantized @ ", args.bit, "b accuracy (first try): %.02f%%" % acc)
        df.loc[1,'acc_' + str(args.bit)] = acc

        model = nemo.transform.dequantize_pact(model)
        model = nemo.transform.quantize_pact(model, dummy_input=torch.randn((1,1,28,28)).to(device))
        model.change_precision(bits=1, min_prec_dict=precision)
        acc = test(model, device, test_loader)
        print("\nFakeQuantized @ mixed-precision accuracy after dequant and quant operation: %.02f%%" % acc)

        #with model.statistics_act():
        #    _ = test(model, device, test_loader)
        #model.reset_alpha_act()
        #acc = test(model, device, test_loader)
        #print("\nFakeQuantized @ ", args.bit,"b accuracy (calibrated): %.02f%%" % acc)


        # Now the accuracy is substantially the same as the initial one! This is what we expect to see using a very conservative quantization scheme with 16 bits. Due to the way that NEMO implements the *FakeQuantized* stage, it is very easy to explore what happens by imposing a stricter or mixed precision quantization scheme. The number of bits we can use is very free: we can even set it to "fractionary" values if we want, which corresponds to intermediate $\varepsilon$ *quantum* sizes with respect to the nearest integers. For example, let's force `conv1`, `conv2` and `fc1` to be 7 bits, `fc2` to use only 3 bits for its parameters, and all activations to be 8-bit.

    overall_loss = Q_err_list('overall_loss')
    q_error_list_wt = Q_err_list('q_error_list_wt')
    q_error_list_act = Q_err_list('q_error_list_act')
    if args.qat:
        fig = plt.figure()
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch)
            train_loss = train(model, device, train_loader, optimizer, epoch)
            overall_loss.update(train_loss)
            
            _, q_err_wt, q_err_act = qerror_loss(model)
            q_error_list_wt.update(q_err_wt.cpu().detach().numpy())
            q_error_list_act.update(q_err_act.cpu().detach().numpy())
            prec = test(model, device, test_loader, verbose=True)
            #if epoch % 5 == 0:
            #    model = act_alphasetting(model, device, test_loader)
        plt.plot(overall_loss.tolist)
        plt.ylabel("Overall loss")
        plt.xlabel("epochs")
        fig.savefig('./output/loss_' + str(args.bit) + '.png')
        fig = plt.figure()
        plt.plot(q_error_list_wt.tolist, label="q_error_list_wt")
        plt.plot(q_error_list_act.tolist, label="q_error_list_act")
        plt.ylabel("Q err loss")
        plt.xlabel("epochs")
        plt.legend(loc="upper right")
        fig.savefig('./output/q_error_' + str(args.bit) + '.png')
        
    ## Add prec to df
    df.loc[2,'acc_' + str(args.bit)] = prec
    with model.statistics_act():
        _ = test(model, device, test_loader)
    model.unset_statistics_act()
    model.reset_alpha_act()
    acc = test(model, device, test_loader)
    print("\nFakeQuantized @ ", args.bit,"b accuracy (calibrated): %.02f%%" % acc)
    df.loc[3,'acc_' + str(args.bit)] = acc

    # save checkpoint using NEMO's dedicated function
    nemo.utils.save_checkpoint(model, None, 0, checkpoint_name='mnist_fq_mixed')
    # load it back (just for fun!) with PyTorch's one
    checkpoint = torch.load('checkpoint/mnist_fq_mixed.pth')
    # pretty-print the precision dictionary
    import json
    print(json.dumps(checkpoint['precision'], indent=2))


    model.fold_bn()
    model.reset_alpha_weights()
    acc = test(model, device, test_loader)
    print("\nFakeQuantized @ mixed-precision (folded) accuracy: %.02f%%" % acc)
    df.loc[4,'acc_' + str(args.bit)] = acc

    model = nemo.transform.bn_to_identity(model) # necessary because folding does not physically remove BN layers
    acc = test(model, device, test_loader)
    print("\nQuantizedDeployable @ mixed-precision accuracy before qd stage: %.02f%%" % acc)
    model.qd_stage(eps_in=1./255)
    acc = test(model, device, test_loader)
    print("\nQuantizedDeployable @ mixed-precision accuracy: %.02f%%" % acc)
    df.loc[5,'acc_' + str(args.bit)] = acc

    with model.statistics_act():
        _ = test(model, device, test_loader)
    model.unset_statistics_act()
    model.reset_alpha_act()
    acc = test(model, device, test_loader)
    print("\nQuantizedDeployable @ ", args.bit,"b accuracy (calibrated): %.02f%%" % acc)
    df.loc[6,'acc_' + str(args.bit)] = acc

    # The *QuantizedDeployable* network is accurate only in the sense that the operations keep all quantization assumptions. It is not, however, bit-accurate with respect to deployment on an integer-only hardware platform. To get that level of accuracy, we have to transform the network to the last stage: ***IntegerDeployable***. This is done, again, by a high-level `id_stage` method. Most of the book-keeping operations have already been performed at this stage and the only differences between the QD and ID networks are due to numerical corner cases (e.g., some numbers get quantized in a certain "box" in QD and in the adjacent ones in ID due to float32 rounding -- and these errors can propagate).
    # 
    # At this stage, the network can essentially "forget" about the quantum and only work on integer images in all nodes. This means that all weights and activations are replaced by integers! The next cells shows the transformation and tests the final integer network... if we remember that now also test data has to be represented in an integer format.


    # In this case, BatchNorm folding does not hit the accuracy of the network at all -- in other cases, this operation can be more disruptive and you may prefer to keep BatchNorm operations.

    # In this case, BatchNorm folding does not hit the accuracy of the network at all -- in other cases, this operation can be more disruptive and you may prefer to keep BatchNorm operations.

    model.id_stage()
    print(model)
    acc = test(model, device, test_loader)
    print("\nIntegerDeployable @ mixed-precision accuracy: %.02f%%" % acc)
    df.loc[7,'acc_' + str(args.bit)] = acc


    # Terrible, 11% is basically random! What did not work? We forgot that also the input data has to be represented as an integer: our `test` function, right now,. Let us redefine our test function so that it considers also the integer case:
    acc = test_with_integer(model, device, test_loader, integer=True)
    print("\nIntegerDeployable @ mixed-precision accuracy (for real): %.02f%%" % acc)
    df.loc[8,'acc_' + str(args.bit)] = acc
    #df.to_csv('./output/accuracy_tracesQAT' + str(args.qat) + str(args.bit) + 'bit.csv')
    df.to_csv('./output/accuracy_tracesQAT' + str(args.qat) + '.csv', index=False, header=True)
    
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU() # <== Module, not Function!
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU() # <== Module, not Function!
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 256)
        self.fcrelu1 = nn.ReLU() # <== Module, not Function!
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x) # <== Module, not Function!
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x) # <== Module, not Function!
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fcrelu1(x) # <== Module, not Function!
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) # <== the softmax operation does not need to be quantized, we can keep it as it is
        return output


#criterion = nn.CrossEntropyLoss()#.cuda()
#Q_error introduced in weight and activation maps is added to the overall loss
def qerror_loss(model):
    loss_qerror = torch.tensor(0.).to(device)
    loss_qerror_wt = torch.tensor(0.).to(device)
    loss_qerror_act = torch.tensor(0.).to(device)
    
    for m in model.modules():
        if isinstance(m, PACT_Conv2d):
            loss_qerror     = loss_qerror.to(device)        + m.q_err_wt.to(device) + m.q_err_act.to(device)
            loss_qerror_wt  = loss_qerror_wt.to(device)     + m.q_err_wt.to(device) 
            loss_qerror_act = loss_qerror_act.to(device)    + m.q_err_act.to(device)
    return loss_qerror, loss_qerror_wt, loss_qerror_act
    
def plot_qerror_loss(model):
    fig = plt.figure()
    for m in model.modules():
        if isinstance(m, PACT_Conv2d):
            plt.plot(m.q_err_list_wt.tolist.numpy().squeeze())
    #plt.show()
    fig.savefig('./first.png')

class Q_err_list(object):
    def __init__(self, name):
        self.name = name
        self.elements = []
    def update(self, val):
        self.elements.append(val)
    @property
    def tolist(self):
        return self.elements

        
# convenience class to keep track of averages
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.).to(device)
        self.n = torch.tensor(0.).to(device)
    def update(self, val):
        self.sum += val.to(device)
        self.n += 1
    @property
    def avg(self):
        return self.sum / self.n

# Then we define the training and testing functions (MNIST has no validation set). These are essentially identical to regular PyTorch code, with only one difference: testing (and validation) functions have a switch to support the production of non-negative integer data. This is important to test the last stage of quantization, i.e., *IntegerDeployable*. Of course, this change might also be effectively performed inside the data loaders; in this example, we use standard `torchvision` data loaders for MNIST.
def train(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    train_loss = Metric('train_loss')
    loss_list = []
    with tqdm(total=len(train_loader),
          desc='Train Epoch     #{}'.format(epoch + 1),
          disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_output = F.nll_loss(output, target)
            loss_qerror, _, _ = qerror_loss(model)
            loss = loss_output + loss_qerror  #Remove loss_qerror to ignore quantization error

            loss.backward()
            optimizer.step()
            train_loss.update(loss)
            t.set_postfix({'loss': train_loss.avg.item()})
            t.update(1)
    return train_loss.avg.item()

def test(model, device, test_loader, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = Metric('test_acc')
    with tqdm(total=len(test_loader),
          desc='Test',
          disable=not verbose) as t:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                test_acc.update((pred == target.view_as(pred)).float().mean())
                t.set_postfix({'acc' : test_acc.avg.item() * 100. })
                t.update(1)
    test_loss /= len(test_loader.dataset)
    return test_acc.avg.item() * 100.

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [5, 10, 15, 20]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

def act_alphasetting(model, device, test_loader):
    with model.statistics_act():
        _ = test(model, device, test_loader)
    model.reset_alpha_act()
    acc = test(model, device, test_loader)
    print("\nFakeQuantized @ ", args.bit,"b accuracy (calibrated): %.02f%%" % acc)
    return model



def test_with_integer(model, device, test_loader, verbose=True, integer=False):
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = Metric('test_acc')
    with tqdm(total=len(test_loader),
          desc='Test',
          disable=not verbose) as t:
        with torch.no_grad():
            for data, target in test_loader:
                if integer:      # <== this is different from the previous version
                    data *= 255  # <== of test function!
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                test_acc.update((pred == target.view_as(pred)).float().mean())
                t.set_postfix({'acc' : test_acc.avg.item() * 100. })
                t.update(1)
    test_loss /= len(test_loader.dataset)
    return test_acc.avg.item() * 100.


if __name__=='__main__':
    main()