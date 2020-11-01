import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import util
import pandas as pd
import numpy as np


def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)
    
def reduce_resolution_CT(inpt_exam, outpt_size):
    if len(inpt_exam) < outpt_size:
        return(inpt_exam)
    result = pd.DataFrame(columns = inpt_exam.columns)
    temp = inpt_exam.reset_index().drop(['index'], axis = 1)
    fact = int((len(inpt_exam)*10/outpt_size))
    rm_count = fact - 10
    #remove rm_count element from every fact batch
    step = int(len(inpt_exam)/fact)
    for i in range(step):
        batch_t = temp[temp.index <= (i + 1)*fact]
        batch = batch_t[batch_t.index > i*fact]
        #rm_batch = batch[batch.pe_present_on_image == 0]
        rm_batch = batch
        if len(rm_batch) >= rm_count:
            rm_idx = random.sample(list(rm_batch.index), rm_count)
            batch.drop(rm_idx, axis = 0, inplace = True)
            #print(rm_idx)
        else:
            batch.drop(rm_batch.index, axis = 0, inplace = True)
            rm_idx = random.sample(list(batch.index), rm_count - len(rm_batch))
            batch.drop(rm_idx, axis = 0, inplace = True)
            #print(rm_idx)
        result = pd.concat([result, batch]).reset_index().drop(['index'], axis = 1)
    if len(result) == outpt_size -1:
        result = pd.concat([result, result[len(result)-1:]]).reset_index().drop(['index'], axis = 1)
    return(result)
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        crit = nn.BCELoss()
        BCE_loss = crit(inputs, targets) + 1e-7
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def form_left_dataloaders(pleft=0.25, nleft = 0.25, Num_max = 3500, NM_test = 300):
    tab = pd.read_csv('/media/aous/Samsung_T5/mtrain_sliced_48.csv')
    repart = list(tab['StudyInstanceUID'].unique())
    test_exams = repart[:1000]
    train_exams = repart[1000:]
    traindf = tab[tab['StudyInstanceUID'].isin(train_exams)]
    testdf = tab[tab['StudyInstanceUID'].isin(test_exams)]

    classes = pd.read_csv('/media/aous/Samsung_T5/classes.csv')
    train_classes = classes[classes.slice.isin(traindf['slice'].unique())]
    test_classes = classes[classes.slice.isin(testdf['slice'].unique())]
    
    L = list(np.random.choice(train_classes[train_classes.leftsided_pe == 1]['slice'].unique(), int(Num_max*pleft), replace=False))
    print(int(Num_max*pleft), "left")

    Negative_Exams = train_classes[train_classes.PE == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - nleft - pleft)), replace=False))
    print(int(Num_max*(1 - nleft - pleft)), "negative")

    Positive_Exams = train_classes[train_classes.PE == 1][train_classes.leftsided_pe == 0]['slice'].unique()
    np.random.shuffle(Positive_Exams)
    L += list(np.random.choice(Positive_Exams, int(Num_max*nleft), replace=False))
    print(len(L), "all")

    Num_max = NM_test 
    np.random.shuffle(L)
    new_traindf = traindf[traindf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    L = list(np.random.choice(test_classes[test_classes.leftsided_pe == 1]['slice'].unique(), int(Num_max*pleft), replace=False))
    

    Negative_Exams = test_classes[test_classes.PE == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - nleft - pleft)), replace=False))

    Positive_Exams = test_classes[test_classes.PE == 1][test_classes.leftsided_pe == 0]['slice'].unique()
    np.random.shuffle(Positive_Exams)
    L += list(np.random.choice(Positive_Exams, int(Num_max*nleft), replace=False))

    np.random.shuffle(L)
    valdf = testdf[testdf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    return(new_traindf, testdf, valdf)
def form_chronic_dataloaders(pch=0.1, nch = 0.1, pacc = 0.1, Num_max = 3500, NM_test = 300):
    tab = pd.read_csv('/media/aous/Samsung_T5/mtrain_sliced_48.csv')
    repart = list(tab['StudyInstanceUID'].unique())
    test_exams = repart[:1000]
    train_exams = repart[1000:]
    traindf = tab[tab['StudyInstanceUID'].isin(train_exams)]
    testdf = tab[tab['StudyInstanceUID'].isin(test_exams)]

    classes = pd.read_csv('/media/aous/Samsung_T5/classes.csv')
    train_classes = classes[classes.slice.isin(traindf['slice'].unique())]
    test_classes = classes[classes.slice.isin(testdf['slice'].unique())]
    
    L = list(np.random.choice(train_classes[train_classes.PE == 1][train_classes.chronic_class == 0]['slice'].unique(), int(Num_max*pch), replace=False))
    print(int(Num_max*pch), "positive non chronic")
    
    L = list(np.random.choice(train_classes[train_classes.chronic_class == 1]['slice'].unique(), int(Num_max*nch), replace=False))
    print(int(Num_max*nch), "positive chronic")

    Negative_Exams = train_classes[train_classes.PE == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - nch - pch - pacc)), replace=False))
    print(int(Num_max*(1 - nch - pch - pacc)), "negative")

    Positive_Exams = train_classes[train_classes.chronic_class == 2]['slice'].unique()
    np.random.shuffle(Positive_Exams)
    L += list(np.random.choice(Positive_Exams, int(Num_max*pacc), replace=False))
    print(len(L), "all")

    np.random.shuffle(L)
    new_traindf = traindf[traindf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    Num_max = NM_test 
    
    L = list(np.random.choice(test_classes[test_classes.PE == 1][test_classes.chronic_class == 0]['slice'].unique(), int(Num_max*pch), replace=False))
    
    
    L = list(np.random.choice(test_classes[test_classes.chronic_class == 1]['slice'].unique(), int(Num_max*nch), replace=False))
    

    Negative_Exams = test_classes[test_classes.PE == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - nch - pch - pacc)), replace=False))
    

    Positive_Exams = test_classes[test_classes.chronic_class == 2]['slice'].unique()
    np.random.shuffle(Positive_Exams)
    L += list(np.random.choice(Positive_Exams, int(Num_max*pacc), replace=False))
    print(len(L), "all")

    np.random.shuffle(L)
    valdf = testdf[testdf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    return(new_traindf, testdf, valdf)

def form_central_dataloaders(pleft=0.25, nleft = 0.25, Num_max = 3500, NM_test = 300):
    tab = pd.read_csv('/media/aous/Samsung_T5/mtrain_sliced_48.csv')
    repart = list(tab['StudyInstanceUID'].unique())
    test_exams = repart[:1000]
    train_exams = repart[1000:]
    traindf = tab[tab['StudyInstanceUID'].isin(train_exams)]
    testdf = tab[tab['StudyInstanceUID'].isin(test_exams)]

    classes = pd.read_csv('/media/aous/Samsung_T5/classes.csv')
    train_classes = classes[classes.slice.isin(traindf['slice'].unique())]
    test_classes = classes[classes.slice.isin(testdf['slice'].unique())]
    
    L = list(np.random.choice(train_classes[train_classes.central_pe == 1]['slice'].unique(), int(Num_max*pleft), replace=False))
    print(int(Num_max*pleft), "central")

    Negative_Exams = train_classes[train_classes.PE == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - nleft - pleft)), replace=False))
    print(int(Num_max*(1 - nleft - pleft)), "negative")

    Positive_Exams = train_classes[train_classes.PE == 1][train_classes.central_pe == 0]['slice'].unique()
    np.random.shuffle(Positive_Exams)
    L += list(np.random.choice(Positive_Exams, int(Num_max*nleft), replace=False))
    print(len(L), "all")

    Num_max = NM_test 
    np.random.shuffle(L)
    new_traindf = traindf[traindf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    L = list(np.random.choice(test_classes[test_classes.central_pe == 1]['slice'].unique(), int(Num_max*pleft), replace=False))
    

    Negative_Exams = test_classes[test_classes.PE == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - nleft - pleft)), replace=False))

    Positive_Exams = test_classes[test_classes.PE == 1][test_classes.central_pe == 0]['slice'].unique()
    np.random.shuffle(Positive_Exams)
    L += list(np.random.choice(Positive_Exams, int(Num_max*nleft), replace=False))

    np.random.shuffle(L)
    valdf = testdf[testdf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    return(new_traindf, testdf, valdf)

def form_rvlv_dataloaders(pleft=0.25, Num_max = 3500, NM_test = 300):
    tab = pd.read_csv('/media/aous/Samsung_T5/mtrain_rvlv_70.csv')
    repart = list(tab['StudyInstanceUID'].unique())
    test_exams = repart[:1000]
    train_exams = repart[1000:]
    traindf = tab[tab['StudyInstanceUID'].isin(train_exams)]
    testdf = tab[tab['StudyInstanceUID'].isin(test_exams)]

    
    L = list(np.random.choice(traindf[traindf.rv_lv_ratio_gte_1 == 1]['slice'].unique(), int(Num_max*pleft), replace=False))
    print(int(Num_max*pleft), "rv >= lv")

    Negative_Exams = traindf[traindf.rv_lv_ratio_gte_1 == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - pleft)), replace=False))
    print(int(Num_max*(1 - pleft)), "negative")
    
    np.random.shuffle(L)
    new_traindf = traindf[traindf.slice.isin(L)].reset_index().drop(['index'], axis = 1)

    Num_max = NM_test
    
    L = list(np.random.choice(testdf[testdf.rv_lv_ratio_gte_1 == 1]['slice'].unique(), int(Num_max*pleft), replace=False))
    print(int(Num_max*pleft), "rv >= lv")

    Negative_Exams = testdf[testdf.rv_lv_ratio_gte_1 == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - pleft)), replace=False))
    print(int(Num_max*(1 - pleft)), "negative")

    np.random.shuffle(L)
    valdf = testdf[testdf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    return(new_traindf, testdf, valdf)

def form_detection_dataloaders(positives=0.25, Num_max = 3500, NM_test = 300):
    tab = pd.read_csv('/media/aous/Samsung_T5/mtrain_sliced_48.csv')
    repart = list(tab['StudyInstanceUID'].unique())
    test_exams = repart[:1000]
    train_exams = repart[1000:]
    traindf = tab[tab['StudyInstanceUID'].isin(train_exams)]
    testdf = tab[tab['StudyInstanceUID'].isin(test_exams)]

    classes = pd.read_csv('/media/aous/Samsung_T5/classes.csv')
    train_classes = classes[classes.slice.isin(traindf['slice'].unique())]
    test_classes = classes[classes.slice.isin(testdf['slice'].unique())]
    
    L = list(np.random.choice(train_classes[train_classes.PE == 1]['slice'].unique(), int(Num_max*positives), replace=False))
    print(int(Num_max*positives), "positive")

    Negative_Exams = train_classes[train_classes.PE == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - positives)), replace=False))
    print(int(Num_max*(1 - positives)), "negative")

    Num_max = NM_test 
    np.random.shuffle(L)
    new_traindf = traindf[traindf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    L = list(np.random.choice(test_classes[test_classes.PE == 1]['slice'].unique(), int(Num_max*positives), replace=False))
    

    Negative_Exams = test_classes[test_classes.PE == 0]['slice'].unique()
    np.random.shuffle(Negative_Exams)
    L += list(np.random.choice(Negative_Exams, int(Num_max*(1 - positives)), replace=False))

    np.random.shuffle(L)
    valdf = testdf[testdf.slice.isin(L)].reset_index().drop(['index'], axis = 1)
    
    return(new_traindf, testdf, valdf)

def weighted_dataset(Positives, size, test_size):
    tab = pd.read_csv('/media/aous/Samsung_T5/mtrain.csv')
    repart = list(tab['StudyInstanceUID'].unique())
    test_exams = repart[:1000]
    train_exams = repart[1000:]
    traindf = tab[tab['StudyInstanceUID'].isin(train_exams)]
    testdf = tab[tab['StudyInstanceUID'].isin(test_exams)]
    
    p_num = int(size*Positives)
    n_num = size - p_num
    print("positives", p_num)
    print("negatives", n_num)
    P = traindf[traindf.pe_present_on_image == 1].reset_index()
    P = P.drop(['index'], axis =1)
    N = traindf[traindf.pe_present_on_image == 0].reset_index()
    N = N.drop(['index'], axis =1)
    print(P['SOPInstanceUID'].nunique())
    L = list(np.random.choice(P['SOPInstanceUID'].unique(), p_num, replace = False))
    L += list(np.random.choice(N['SOPInstanceUID'].unique(), n_num, replace = False))
    np.random.shuffle(L)
    new_traindf = traindf[traindf.SOPInstanceUID.isin(L)].reset_index().drop(['index'], axis = 1)
    
    p_num = int(test_size*Positives)
    n_num = test_size - p_num
    P = testdf[testdf.pe_present_on_image == 1].reset_index()
    P = P.drop(['index'], axis =1)
    N = testdf[testdf.pe_present_on_image == 0].reset_index()
    N = N.drop(['index'], axis =1)
    L = list(np.random.choice(P['SOPInstanceUID'].unique(), p_num, replace = False))
    L += list(np.random.choice(N['SOPInstanceUID'].unique(), n_num, replace = False))
    np.random.shuffle(L)
    valdf = testdf[testdf.SOPInstanceUID.isin(L)].reset_index().drop(['index'], axis = 1)
    return(new_traindf, testdf, valdf)