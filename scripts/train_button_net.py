import os, argparse, logging
import sys
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import skimage

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from tqdm import trange, tqdm
from ipdb import set_trace as st
from models.classifiers import AttributeClassifier
from utils.builders import ButtonDataset, GaussianAdditiveNoiseTransform
from utils.utils import weight_init, set_gpu_mode, zeros, get_numpy
from torchvision.transforms import ToTensor, Compose
### Set GPU visibility
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1, 2"  # Set this for adequate GPU usage

### Set Global Parameters
# _LOSS = nn.NLLLoss
_LOSS = nn.BCELoss
_RECON_LOSS = nn.MSELoss
ROOT_DIR = '/home/msieb/git/button-learning/env_static/logs'

### Helper functions
def apply(func, M):
    """Applies a function over a batch (PyTorch as of now has no support for arbitrary function calls on batches)
    """

    tList = [func(m) for m in torch.unbind(M, dim=0) ]
    res = torch.stack(tList, dim=0)
    return res

def compute_acc(pred_labels, gt_labels):
    pred_labels = get_numpy(pred_labels)
    gt_labels = get_numpy(gt_labels)
    n_correct = np.sum(pred_labels == gt_labels)
    return 1.0 * n_correct / (gt_labels.shape[1] * gt_labels.shape[0])

def compute_recall(pred_labels, gt_labels):
    pred_labels = get_numpy(pred_labels)
    gt_labels = get_numpy(gt_labels)
    n_correct_pos = np.sum(pred_labels * gt_labels)
    # return 1.0 * n_correct_pos / np.sum(gt_labels)
    return -1

def MI(Xs, Ys):
    '''
    Computes empirical mutual information of batches of pairwise r.v.'s
    Xs, Ys are in shape B by N by D
    B is number of pairwise r.v's
    N is number of samples per r.v.
    D is the dimension of r.v.

    output is dimension B by 1
    '''
    
    B, N, D = Xs.shape
    
    idx_fast, idx_slow = [], []
    for i in range(D):
        for j in range(D):
            idx_slow.append(i)
            idx_fast.append(j)
    Xs_slow = Xs[:, :, idx_fast]
    Ys_fast = Ys[:, :, idx_slow]

    joint_ps = torch.mean(Xs_slow * Ys_fast, dim=1)
    X_ps = torch.mean(Xs, dim=1)
    Y_ps = torch.mean(Ys, dim=1)

    mi = torch.sum( joint_ps * \
        (
        torch.log2(torch.clamp(joint_ps, 1e-7, 1)) \
        - torch.log2(torch.clamp(X_ps[:, idx_slow], 1e-7, 1)) \
        - torch.log2(torch.clamp(Y_ps[:, idx_fast], 1e-7, 1)) \
        ),
        dim=1
    )

    return mi


def variation_dist(Xs, Ys):
    '''
    Computes "variation dist" between pairs of sampled r.v.'s
    Xs, Ys are in shape B by N by D
    B is number of pairwise r.v's
    N is number of samples per r.v.
    D is the dimension of r.v.

    output is dimension B by 1
    '''

    Xs_delta_norms = torch.norm(Xs[:, 1:, :] - Xs[:, :-1, :], p=2, dim=1)
    Ys_delta_norms = torch.norm(Ys[:, 1:, :] - Ys[:, :-1, :], p=2, dim=1)

    dists = torch.norm(Xs_delta_norms - Ys_delta_norms, p=2, dim=1)

    return dists


def train(model, loader_tr, loader_t, lr=1e-3, epochs=1000, use_cuda=True):
    """Train model and shows sample results
    
    Parameters
    ----------
    model : torch.nn.Module
        neural network
    loader_tr : Training Dataloader
        Loads training data (images and labels)
    loader_t : Test Dataloader
        Loads test data (images and labels)
    lr : float optional
        Optimizer learning rate (the default is 1e-4)
    epochs : int, optional
        number of training epochs (the default is 1000)
    use_cuda : bool, optional
        whether or not to use GPU (the default is True, which uses GPU)
    
    Returns
    -------
    dictionary of training statistics
        contains metrics such as loss and accuracy
    """

    logs = {
        'loss': {
            'tr': [],
            't': []
        },
        'acc': {
            'tr': [],
            't': []
        },
        'rec': {
            'tr': [],
            't': []
        }

        
    }
    criterion = _LOSS()
    recon_criterion = _RECON_LOSS()
    regularizer = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=lr)
    t_epochs = trange(epochs, desc='{}/{}'.format(0, epochs))
    num_batches_tr = len(loader_tr)
    num_batches_t = len(loader_t)
    dataiter = list(iter(loader_t))
    for e in t_epochs:
        # Train
        loss_tr = 0
        acc_tr = 0
        rec_tr = 0
        t_batches = tqdm(loader_tr, leave=False, desc='Train')
  
        for sample in t_batches:
            xb = sample['image']
            yb = sample['label']
            if use_cuda:
                xb = xb.cuda()
                yb = yb.cuda()
            opt.zero_grad()

            pred, emb, emb_logits, recon = model(xb)
            
            entropies = torch.sum(-emb*torch.log2(torch.clamp(emb, 1e-3, 1)), dim=-1)
            entropy_mean = torch.mean(entropies, dim=1)
            entropy_min = torch.min(entropies, dim=1)[0]
            entropy_var = torch.mean(torch.pow(entropies - entropy_mean.unsqueeze(1), 2), dim=1)
            entropy_var_batch = torch.mean(entropy_var)
            entropy_mean_batch = torch.mean(entropy_mean)
            entropy_min_batch = torch.mean(entropy_min)

            entropies_mean_samples = torch.mean(entropies, dim=0)
            entropies_var_samples = torch.mean(torch.pow(entropies - entropies_mean_samples.unsqueeze(0), 2), dim=0)
            entropies_var_samples_batch = torch.mean(entropies_var_samples)

            recon_loss = recon_criterion(xb, recon)
            idx_fast = []
            idx_slow = []
            for i in range(emb.shape[1]):
                for j in range(emb.shape[1]):
                    if i != j:
                        idx_fast.append(j)
                        idx_slow.append(i)
            l1_regularization, l2_regularization = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            for name, param in model.named_parameters():
                if 'Conv2d' in name or 'FC_6a' in name:
                    l1_regularization += torch.norm(param, 1)
                    l2_regularization += torch.norm(param, 2)

            # pairwise_MI = MI(emb[:, idx_slow, :].permute(1, 0, 2), emb[:, idx_fast, :].permute(1, 0, 2))
            # pairwise_MI_mean = torch.mean(pairwise_MI)
            # pairwise_MI_max = torch.max(pairwise_MI)

            # variation_dists = variation_dist(emb[:, idx_slow, :].permute(1, 0, 2), emb[:, idx_fast, :].permute(1, 0, 2))
            # variation_dists_mean = torch.mean(variation_dists)

            # loss = criterion(pred, yb) + alpha*regularizer(emb, torch.zeros(emb_logits.shape).cuda())
            # loss = criterion(pred, yb) 
            losses = [
                criterion(pred, yb),
                # -0.1 * entropy_mean_batch,
                # -0.1 * entropy_var_batch,
                # 0.2 * entropy_min_batch,
                # 0.2 * entropies_var_samples_batch,
                0.15 * regularizer(emb_logits, torch.zeros(emb_logits.shape).cuda()),
                1. * recon_loss,
                # torch.sum(entropies),
                0.1 * l2_regularization
            ]
            loss = sum(losses)

            labels_pred = torch.round(pred)
            acc = compute_acc(labels_pred, yb)
            rec = compute_recall(labels_pred, yb)

            loss_tr += loss
            acc_tr += acc
            rec_tr += rec
           

            loss.backward()
            opt.step()

            t_batches.set_description('Train: {:.10f}, {:.2f}, {:.2f}'.format(loss_tr, acc_tr, rec_tr))
            t_batches.update()
        

        loss_tr /= num_batches_tr
        acc_tr /= num_batches_tr
        rec_tr /= num_batches_tr


        ## TODO implement validation
        # Eval on test
        loss_t = 0
        acc_t = 0
        rec_t = 0
        for sample in tqdm(loader_t, leave=False, desc='Eval'):
            xb = sample['image']
            yb = sample['label']           
            if use_cuda:
                xb = xb.cuda()
                yb = yb.cuda()
            pred, emb, emb_logits, recon = model(xb)
            # TODO: not real loss
            loss = criterion(pred, yb)# + alpha*regularizer(emb, torch.zeros(emb_logits.shape).cuda())

            labels_pred = torch.round(pred)
            acc = compute_acc(labels_pred, yb)
            rec = compute_recall(labels_pred, yb)

            loss_t += loss
            acc_t += acc
            rec_t += rec

            if (e+1) % 90 == 0:
                print('embedding: \n{}\ngt encoding: \n{}'.format(get_numpy(emb), get_numpy(sample['encoding'].view(-1, 2, 3))))
                # print('pred: \n{}\ngt : \n{}'.format(get_numpy(labels_pred), get_numpy(sample['label']  )))
                plt.figure(0)
                plt.imshow(np.transpose((get_numpy(xb[0])*255).astype(np.uint8), (1,2,0)))
                plt.figure(1)
                plt.imshow(np.transpose((get_numpy(recon[0])*255).astype(np.uint8), (1,2,0)))
                plt.show()
                import ipdb; ipdb.set_trace()
        loss_t /= num_batches_t
        acc_t /= num_batches_t
        rec_t /= num_batches_t
        
        t_epochs.set_description('{}/{} | Tr {:.2f}, {:.2f}, {:.2f}. T {:.2f}, {:.2f}, {:.2f}'.format(e, epochs, loss_tr, acc_tr, rec_tr, loss_t, acc_t, rec_t))
        t_epochs.update()
        print('epoch: ', e)
        print('train_loss: ', loss_tr)
        print('test_loss: ', loss_t)
        logs['loss']['tr'].append(loss_tr)
        logs['acc']['tr'].append(acc_tr)
        logs['rec']['tr'].append(rec_tr)
        logs['loss']['t'].append(loss_t)
        logs['acc']['t'].append(acc_t)
        logs['rec']['t'].append(rec_t)

        print('-'*10)
    return logs

def create_model(args, use_cuda=True):
    """Creates neural network model, loading from checkpoint of provided
    
    Parameters
    ----------
    args : variable function arguments
        see parser in main for details
    use_cuda : bool, optional
        whether or not to use GPU (the default is True, which uses GPU)
    
    Returns
    -------
    torch.nn.Module
        contains EENet model
    """

    # model = BinaryEncoder(emb_dim=6)
    model = AttributeClassifier(emb_dim=3, num_attr=2)

    if args.load_model:
        model_path = os.path.join(
            args.model_path,
        )
        # map_location allows us to load models trained on cuda to cpu.
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if use_cuda:
        model = model.cuda()
    return model

if __name__ == '__main__':
    """Parses arguments, creates dataloaders for training and test data, sets up model and logger, and trains network
    """

    ### Setting up parser, logger and GPU params
    set_gpu_mode(True)
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', '-t', type=float, default=0.2)
    parser.add_argument('--epochs', '-e', type=int, default=1000)
    parser.add_argument('--learning_rate', '-r', type=float, default=1e-3)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--root_dir', type=str, default=ROOT_DIR)
    parser.add_argument('-sf', '--load_data_and_labels_from_same_folder', action='store_true')
    args = parser.parse_args()

    print('\n')
    logging.info('Make sure you provided the correct GPU visibility in line 24 depending on your system !')
    logging.info('Loading {}'.format(args.root_dir))
    logging.info('Processing Data')
    
    ### Create dataset
    dataset = ButtonDataset(root_dir=args.root_dir, transform=Compose([
        GaussianAdditiveNoiseTransform(0, 0.02)
    ]))

    # Split dataset in training and test set
    n = len(dataset)
    n_test = int( n * .2 )  # number of test/val elements
    n_train = n - 2 * n_test
    dataset_tr, dataset_t, dataset_val = train_set, val_set, test_set = random_split(dataset, (n_train, n_test, n_test))
    loader_tr = DataLoader(dataset_tr, batch_size=10,
                        shuffle=True, num_workers=4)
    loader_t = DataLoader(dataset_t, batch_size=1, shuffle=True)                       
    
    ### Load model
    model = create_model(args)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) # Parallize model if multiple GPUs are available

    ### Train
    logging.info('Training.')
    logs = train(model, loader_tr, loader_t, lr=args.learning_rate, epochs=args.epochs)
    # TODO save stuff

    # Default into debug mode if training is completed
    import ipdb; ipdb.set_trace()
    exit()

