from time import sleep
import pybullet as p
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


_LOSS = th.nn.BCELoss()
_USE_CUDA = True

if __name__ == "__main__":

    criterion = _LOSS()
    agent = Agent()
    epoch = 100
    opt = optim.Adam(model.parameters(), lr=0.001)
    t_epochs = trange(epochs, desc='{}/{}'.format(0, epochs))


    img = plt.imread('/home/max/git/button-learning/experiments/button/56/rgb/00000.png')

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)

    boxes = []

    for i in range(4):
        for j in range(4):
            h = 100
            w = 100
            coords = (10+i*130, 530-j*130)
            boxes.append(img[coords[1]:coords[1]+h, coords[0]:coords[0]+h])
            rect = patches.Rectangle(coords,w,h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            axs[i, j].imshow(img[coords[1]:coords[1]+h, coords[0]:coords[0]+h])
    plt.show()

    for epoch in t_epochs:
        # Train
        loss_tr = 0
        acc_tr = 0
        # t_batches = tqdm(loader_tr, leave=False, desc='Train')

        # for sample in t_batches:
        
        # boxes = sample['boxes']
        # target_classes = sample['label']
        boxes = torch.Tensor(boxes).float()
        target_classes = torch.Tensor([1, 4, 8])

        if _USE_CUDA:
            boxes = boxes.cuda()
            target_classes = target_classes.cuda()

        optimizer.zero_grad()

        sigmoid_probs = agent.predict(boxes, classifier_type='rgb')


        loss = criterion(sigmoid_probs, target_classes)
        loss.backward()
        optimizer.step()

        t_batches.update()
