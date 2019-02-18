from time import sleep
import numpy as np
import math
from abc import ABCMeta, abstractmethod
import random
import torch as th
import torch.optim as optim
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.util import create_ranges_divak
from agent import Agent

from pdb import set_trace as st

_LOSS = th.nn.BCEWithLogitsLoss()

_USE_CUDA = True

if __name__ == "__main__":

    criterion = _LOSS
    agent = Agent()
    epochs = 1000
    optimizer = optim.Adam(agent.rgb_classifier.parameters(), lr=0.001)
    t_epochs = trange(epochs, desc='{}/{}'.format(0, epochs))


    img = plt.imread('./experiments/button/56/rgb/00000.png')

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)

    boxes = []

    for i in range(4):
        for j in range(4):
            h = 100
            w = 100
            coords = (10+i*130, 530-j*130)
            box = img[coords[1]:coords[1]+h, coords[0]:coords[0]+h]
            box = np.transpose(box, (2, 0, 1))
            boxes.append(box)
            rect = patches.Rectangle(coords,w,h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            axs[i, j].imshow(img[coords[1]:coords[1]+h, coords[0]:coords[0]+h])
#    plt.show()

    for epoch in t_epochs:
        # Train
        loss_tr = 0
        acc_tr = 0
        # t_batches = tqdm(loader_tr, leave=False, desc='Train')

        # for sample in t_batches:
        
        # boxes = sample['boxes']
        # target_classes = sample['label']
        input_boxes = th.Tensor(boxes).float()
        target_inds = [2, 7, 14]
        target_classes = th.zeros(len(input_boxes))
        target_classes[target_inds] = 1
        target_classes.unsqueeze_(1)
        
        if _USE_CUDA:
            input_boxes = input_boxes.cuda()
            target_classes = target_classes.cuda()

        optimizer.zero_grad()

        probs_logits = agent.predict(input_boxes, classifier_type='rgb')
        sigmoid_probs = th.nn.Sigmoid()(probs_logits)
        #loss = criterion(probs_logits, target_classes)
        if epoch % 3 == 0:
            probs_max_ind = np.random.choice(len(boxes), 1)
        else:
            probs_max_ind = th.max(probs_logits, 0)[1].long()
        if epoch == 0:
            probs_max_ind = 7
        #print('logit: {}, target: {}'.format(probs_logits[probs_max_ind, ...], target_classes[probs_max_ind, ...]))
        loss = criterion(probs_logits[probs_max_ind, ...], target_classes[probs_max_ind, ...])
        loss.backward()
        
        optimizer.step()
        print('loss: {}'.format(loss.cpu().detach().numpy()))
        if epoch % 3 == 0:
            print('---')
            #print(sigmoid_probs.cpu().detach().numpy())
            print(np.where(sigmoid_probs.cpu().detach().numpy() > 0.8)[0])
            print('---')
            st()
        #t_batches.update()
