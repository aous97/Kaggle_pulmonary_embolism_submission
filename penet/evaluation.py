import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import util
import pandas as pd
import numpy as np
from tqdm import tqdm

def TP(predicted, labels, positive_label = 1):
    return(torch.logical_and((predicted == labels) , (labels == positive_label)).sum().item())
def TN(predicted, labels, positive_label = 1):
    return(torch.logical_and((predicted != positive_label) , (labels != positive_label)).sum().item())
def FP(predicted, labels, positive_label = 1):
    return(torch.logical_and((predicted != labels) , (labels == positive_label)).sum().item())
def FN(predicted, labels, positive_label = 1):
    return(torch.logical_and((predicted != labels) , (labels != positive_label)).sum().item())
def Fscore(TP, FP, TN, FN):
    precision = TP/(TP + FP + 0.0001)
    recall = TP/(TP + FN + 0.0001)
    return(precision, recall, 2*precision*recall/(precision + recall + 0.0001))

def evaluate(net, test_loader):
    total = 0
    correct = 0
    net.eval()
    net.cuda()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, (inputs, target) in tqdm(enumerate(test_loader)):
        inputs = inputs.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        outputs = net(inputs)
        #print(outputs, target)
        #if i == 4:
        #    ok
        _, pred = torch.max(outputs.data, 1)
        #predicted = (outputs>0.5).float().view(target.size(0))
        detection = pred
        
        tp += TP(detection, target)
        tn += TN(detection, target)
        fp += FP(detection, target)
        fn += FN(detection, target)
        correct += tp + tn
        total += tp + tn + fp + fn
        
    net.train()
    f0 = Fscore(tp, fp, tn, fn)
    acc= correct / total
    print("TP: ", tp, "TN: ", tn, "FP: ", fp, "FN: ", fn)
    print(acc)
    return(f0)

def evaluate_per_class(net, test_loader, classe):
    total = 0
    correct = 0
    net.eval()
    net.cuda()
    tp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    for i, (inputs, target) in tqdm(enumerate(test_loader)):
        inputs = inputs.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        outputs = net(inputs)
        #print(outputs, target)
        #if i == 4:
        #    ok

        predicted = outputs.float()
        detection = predicted[:, 0]
        location1 = predicted[:, 1]
        tp[0] += TP(detection, target[:, 0])
        tn[0] += TN(detection, target[:, 0])
        fp[0] += FP(detection, target[:, 0])
        fn[0] += FN(detection, target[:, 0])
        
        if classe != 4:
            tp[1] += TP(location1, target[:, classe])
            tn[1] += TN(location1, target[:, classe])
            fp[1] += FP(location1, target[:, classe])
            fn[1] += FN(location1, target[:, classe])
        else:
            _, chronic = torch.max(outputs[: , 1:].data, 1)
            location = chronic
            tp[1] += TP(location, target[:, classe], 0)
            tn[1] += TN(location, target[:, classe], 0)
            fp[1] += FP(location, target[:, classe], 0)
            fn[1] += FN(location, target[:, classe], 0)
            tp[2] += TP(location, target[:, classe], 1)
            tn[2] += TN(location, target[:, classe], 1)
            fp[2] += FP(location, target[:, classe], 1)
            fn[2] += FN(location, target[:, classe], 1)
            tp[3] += TP(location, target[:, classe], 2)
            tn[3] += TN(location, target[:, classe], 2)
            fp[3] += FP(location, target[:, classe], 2)
            fn[3] += FN(location, target[:, classe], 2)
            total += target.size(0)
            correct += (chronic == target[:, classe]).sum().item()
        
    net.train()
    f0 = Fscore(tp[0], fp[0], tn[0], fn[0])
    f1, f2 , f3 = 0, 0, 0
    print(tp[0], tn[0], fp[0], fn[0])
    if classe != 4:
        f1 = Fscore(tp[1], fp[1], tn[1], fn[1])
        print(tp[1], tn[1], fp[1], fn[1])
    else:
        f1 = Fscore(tp[1], fp[1], tn[1], fn[1])
        print(tp[1], tn[1], fp[1], fn[1])
        f2 = Fscore(tp[2], fp[2], tn[2], fn[2])
        print(tp[2], tn[2], fp[2], fn[2])
        f3 = Fscore(tp[3], fp[3], tn[3], fn[3])
        print(tp[3], tn[3], fp[3], fn[3])
        print(correct/ total, "%")
    return(f0, f1, f2, f3)

def evaluate_per_class_CRE(net, test_loader, classe):
    total = 0
    correct = 0
    net.eval()
    net.cuda()
    tp , tn, fp, fn = 0, 0, 0, 0
    for i, (inputs, target) in tqdm(enumerate(test_loader)):
        inputs = inputs.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        outputs = net(inputs)
        #print(outputs, target)
        #if i == 4:
        #    ok
        
        _, chronic = torch.max(outputs.data, 1)
        location = chronic
        tp += TP(location, target[:, classe])
        tn += TN(location, target[:, classe])
        fp += FP(location, target[:, classe])
        fn += FN(location, target[:, classe])
        total += target.size(0)
        correct += (chronic == target[:, classe]).sum().item()
        
    net.train()
    f0 = Fscore(tp, fp, tn, fn)
    f1 = float(correct)/total
    print("TP: ", tp, "FP: ", fp, "TN: ", tn, "FN: ", fn) 
    print(correct, total)
    return(f0, f1)