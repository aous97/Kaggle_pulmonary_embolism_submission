import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import cv2
import time
import pandas as pd
import torchvision
import torch
import torchvision.transforms as transforms
import sys


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from tqdm import tqdm_notebook, tnrange

import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision import datasets
from tqdm.notebook import *
from torchcontrib.optim import SWA

import time
import matplotlib.pyplot as plt


def train_detection_only(net, optimizer, train_loader, criterion1, criterion2, alpha = 1, beta = 1,  n_epoch = 2,
          train_acc_period = 5, classe = 2,
          test_acc_period = 5, detection = False,
          cuda=True, visualize = True, scheduler = None):
    loss_train = []
    loss_test = []
    total = 0
    if cuda:
        net = net.cuda()
    for epoch in tnrange(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        if scheduler != None:
            print(scheduler.get_last_lr())
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            # get the inputs
            if cuda:
                net = net.cuda()
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.clamp(min = 0, max = 1)
            predicted = (outputs>0.5).float()
            if detection:
                loss = criterion1(outputs[:,0].float(), labels[:,0].float())
            else:
                loss = criterion1(outputs[:,0].float(), labels[:,0].float())
                loss += alpha*criterion1(outputs[:,1:4].float(), labels[:,1:4].float())
                loss += beta*criterion2(outputs[:, 4:].float(), labels[:,4])
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            # print statistics
            running_loss = 0.33*loss.item()/labels.size(0) + 0.66*running_loss
            if detection:
                #correct = float((labels[:, 0] == predicted[:, 0]).sum())
                #correct = float(torch.logical_and((labels[:, classe] == predicted[:, 1]), labels[:, classe] == 1).sum())
                #if float((labels[:, classe] == 1).sum()) != 0:
                #    correct /= float((labels[:, classe] == 1).sum())
                _, pred = torch.max(outputs.data, 1)
                correct = float((pred == labels[:, classe]).sum())/labels.size(0)
                #print(pred)
                #print(labels[:, classe])
                #print(outputs.float(), labels.float())
                #print(correct)
                #if i ==10:
                #    ok
            else:
                correct = float((labels[:, 0] == predicted[:, 0]).sum())
                correct += alpha*float((labels[:, 1:4] == predicted[:, 1:4]).sum())
                correct += beta*float((torch.argmax(outputs[:, 4:]) == labels[:, 4]).sum())
                correct /= (1+ alpha*3 + beta)*labels.size(0)
            running_acc = 0.3*correct + 0.66*running_acc
            loss_train.append(running_loss)
            if visualize:
                if i % train_acc_period == train_acc_period-1:
                    #print(labels[:, classe] == 1)
                    #print(predicted[:, 1] == 1)
                    #print('TP: ', torch.logical_and((labels[:, classe] == predicted[:, 1]), labels[:, classe] == 1))
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
                    print('[%d, %5d] acc: %.3f' %(epoch + 1, i + 1, running_acc))
                    running_loss = 0.0
                    total = 0
            inputs = None
            outputs = None
            labels = None
            predicted = None
        if scheduler != None:
            scheduler.step()
      
    if visualize:
        print('Finished Training')
    return(loss_train[-1])

def train_detection_class(net, optimizer, train_loader, criterion1, criterion2, alpha = 1, beta = 1,  n_epoch = 2,
          train_acc_period = 5, classe = 2,
          test_acc_period = 5, detection = False,
          cuda=True, visualize = True, scheduler = None):
    loss_train = []
    loss_test = []
    total = 0
    if cuda:
        net = net.cuda()
    for epoch in tnrange(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        if scheduler != None:
            print(scheduler.get_last_lr())
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            # get the inputs
            if cuda:
                net = net.cuda()
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.clamp(min = 0, max = 1)
            predicted = (outputs>0.5).float()
            if detection:
                loss = alpha*criterion1(outputs[:,0].float(), labels[:,0].float())
                loss += beta*criterion1(outputs[:,1].float(), labels[:,classe].float())
            else:
                loss = criterion1(outputs[:,0].float(), labels[:,0].float())
                loss += alpha*criterion1(outputs[:,1:4].float(), labels[:,1:4].float())
                loss += beta*criterion2(outputs[:, 4:].float(), labels[:,4])
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            # print statistics
            running_loss = 0.33*loss.item()/labels.size(0) + 0.66*running_loss
            if detection:
                #correct = float((labels[:, 0] == predicted[:, 0]).sum())
                #correct = float(torch.logical_and((labels[:, classe] == predicted[:, 1]), labels[:, classe] == 1).sum())
                #if float((labels[:, classe] == 1).sum()) != 0:
                #    correct /= float((labels[:, classe] == 1).sum())
                _, pred = torch.max(outputs.data, 1)
                correct = float((pred == labels[:, classe]).sum())/labels.size(0)
                #print(pred)
                #print(labels[:, classe])
                #print(outputs.float(), labels.float())
                #print(correct)
                #if i ==10:
                #    ok
            else:
                correct = float((labels[:, 0] == predicted[:, 0]).sum())
                correct += alpha*float((labels[:, 1:4] == predicted[:, 1:4]).sum())
                correct += beta*float((torch.argmax(outputs[:, 4:]) == labels[:, 4]).sum())
                correct /= (1+ alpha*3 + beta)*labels.size(0)
            running_acc = 0.3*correct + 0.66*running_acc
            loss_train.append(running_loss)
            if visualize:
                if i % train_acc_period == train_acc_period-1:
                    #print(labels[:, classe] == 1)
                    #print(predicted[:, 1] == 1)
                    #print('TP: ', torch.logical_and((labels[:, classe] == predicted[:, 1]), labels[:, classe] == 1))
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
                    print('[%d, %5d] acc: %.3f' %(epoch + 1, i + 1, running_acc))
                    running_loss = 0.0
                    total = 0
            inputs = None
            outputs = None
            labels = None
            predicted = None
        if scheduler != None:
            scheduler.step()
      
    if visualize:
        print('Finished Training')
    return(loss_train[-1])

def train(net, optimizer, train_loader, criterion1, criterion2, alpha = 1, beta = 1,  n_epoch = 2,
          train_acc_period = 5, classe = 2,
          test_acc_period = 5, detection = False,
          cuda=True, visualize = True, scheduler = None):
    loss_train = []
    loss_test = []
    total = 0
    if cuda:
        net = net.cuda()
    for epoch in tnrange(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        if scheduler != None:
            print(scheduler.get_last_lr())
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            # get the inputs
            if cuda:
                net = net.cuda()
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #outputs = outputs.clamp(min = 0, max = 1)
            predicted = (outputs>0.5).float()
            if detection:
                #loss = 0.2*criterion1(outputs[:,0].float(), labels[:,0].float())
                loss = criterion1(outputs.float(), labels[:,classe])
            else:
                loss = criterion1(outputs[:,0].float(), labels[:,0].float())
                loss += alpha*criterion1(outputs[:,1:4].float(), labels[:,1:4].float())
                loss += beta*criterion2(outputs[:, 4:].float(), labels[:,4])
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            # print statistics
            running_loss = 0.33*loss.item()/labels.size(0) + 0.66*running_loss
            if detection:
                #correct = float((labels[:, 0] == predicted[:, 0]).sum())
                #correct = float(torch.logical_and((labels[:, classe] == predicted[:, 1]), labels[:, classe] == 1).sum())
                #if float((labels[:, classe] == 1).sum()) != 0:
                #    correct /= float((labels[:, classe] == 1).sum())
                _, pred = torch.max(outputs.data, 1)
                correct = float((pred == labels[:, classe]).sum())/labels.size(0)
                #print(pred)
                #print(labels[:, classe])
                #print(outputs.float(), labels.float())
                #print(correct)
                #if i ==10:
                #    ok
            else:
                correct = float((labels[:, 0] == predicted[:, 0]).sum())
                correct += alpha*float((labels[:, 1:4] == predicted[:, 1:4]).sum())
                correct += beta*float((torch.argmax(outputs[:, 4:]) == labels[:, 4]).sum())
                correct /= (1+ alpha*3 + beta)*labels.size(0)
            running_acc = 0.3*correct + 0.66*running_acc
            loss_train.append(running_loss)
            if visualize:
                if i % train_acc_period == train_acc_period-1:
                    #print(labels[:, classe] == 1)
                    #print(predicted[:, 1] == 1)
                    #print('TP: ', torch.logical_and((labels[:, classe] == predicted[:, 1]), labels[:, classe] == 1))
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
                    print('[%d, %5d] acc: %.3f' %(epoch + 1, i + 1, running_acc))
                    running_loss = 0.0
                    total = 0
            inputs = None
            outputs = None
            labels = None
            predicted = None
        if scheduler != None:
            scheduler.step()
      
    if visualize:
        print('Finished Training')
    return(loss_train[-1])