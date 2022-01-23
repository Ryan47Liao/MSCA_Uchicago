#Author:Ryan Date:2022/1/20
from scipy import stats 
import pandas as pd 
import numpy as np 
import copy 

def Sweep_k(A,k,dk,round = 10,eps = 2**-26):
    "Perform sweep algorithm over matrix A on k_th row and col"
    A = np.array(np.around(A,round)) 
    A = A.astype('float64')
    assert A[k,k]!= 0, f'A_kk != 0,but for A_{k}{k} = {A[k,k]}'
    I,J = A.shape
    for i in range(I):
        for j in range(J):
            if i!=k and j!= k:
                A[i,j] = A[i,j] - A[i,k]*A[k,j]/A[k,k] #Step 1
    for i in range(I):
        if i != k:
            A[i,k] = A[i,k] / abs(A[k,k]) #Step 2
    for j in range(J):
        if j != k:
            A[k,j] = A[k,j]/abs(A[k,k]) #Step 3 
    A[k,k] = -1/A[k,k] #Step 4
    return A 

def SWEEP(A,Indexs = None, PRINT = False,round = 3, RETURN_INV = False):
    "Perform SWEEP algorithm REPEATEDLY over matrix A(X.T@X) on specified indexes in the order given"
    Og_A = np.array(A)
    if Indexs is None:
        Indexs = range(A.shape[0]-1)
    for k in Indexs:
        A = Sweep_k(copy.copy(A),k,dk = Og_A[k,k] ) #Upadted 20/01/22
        if PRINT:
            print(k)
            print(np.around(A,round))
    if RETURN_INV:
        return A 
    else:
        return np.around(A[-1,:], round)

def Forward_selection(X,y,alpha = 0.05):
    '''
    X:data with all features
    y:response
    '''
    features = list(X.columns)
    n = X.shape[0]
    if 'Intercept' not in X:
        X.insert(0,'Intercept',1)
        features = ['Intercept'] + features
    X['y'] = y 
    A = np.array(X.T@X) 
    z = A.shape[1] - 1 
    #Start with base model 
    working_list = ['Intercept']
    
    A_s0 = Sweep_k(A,0,A[0,0])
    S0 = A_s0[z,z]
    m0 = len(working_list)
    A_next = A_s0.copy()
    features_temp = features.copy()
    features_temp.remove('Intercept')
    while True:
        candidate_performance = dict()
        m1 = len(working_list) + 1 
        for feature in features_temp:
            #temp_list = working_list.copy()
            #temp_list.append(feature)
            #Get model
            k = features.index(feature)
            A_temp = Sweep_k(A_next.copy(),k = k,dk = A[k,k])
            #Get S1 
            S1 = A_temp[z,z]
            #Get f score
            f_score = ((S0-S1)/(m1-m0)) / (S0/(n-m0))
            #Get p-value 
            p = stats.f.sf(f_score,dfd = n-m0, dfn = m1-m0)
            candidate_performance[feature] = p 
        #print(candidate_performance)
        #In the end, insert the feature corresponding to the least p-value 
        if any(np.array(list(candidate_performance.values())) < alpha):
            feature_selected = min(candidate_performance,key = lambda x: candidate_performance[x])
            working_list.append(feature_selected) #Max if backward
            print(f'Feature {feature_selected} added to the model')
            features_temp.remove(feature_selected)
            k = features.index(feature_selected)
            A_next = Sweep_k(A_next,k = k ,dk = A[k,k])
            S0 = A_next[z,z]
            m0 = len(working_list)
        else:
            return working_list 
    
def Backward_selection(X,y,alpha = 0.05):
    '''
    X:data with all features
    y:response
    '''
    features = list(X.columns)
    n = X.shape[0]
    if 'Intercept' not in X:
        X.insert(0,'Intercept',1)
        features = ['Intercept'] + features
    X['y'] = y 
    A = np.array(X.T@X) 
    z = A.shape[1] - 1 
    #Start with Full model 
    working_list = features.copy()
    
    A_s1 = SWEEP(A, RETURN_INV = True)
    S1 = A_s1[z,z]
    m1 = len(working_list)
    A_next = A_s1.copy()
    features_temp = features.copy()
    features_temp.remove('Intercept')
    while True:
        candidate_performance = dict()
        m0 = len(working_list) - 1 
        for feature in features_temp:
            #Get model
            k = features.index(feature)
            A_temp = Sweep_k(A_next.copy(),k = k,dk = A[k,k]) #Unsweep kth row and col
            #Get S1 
            S0 = A_temp[z,z]
            #Get f score
            f_score = ((S1-S0)/(m0-m1)) / (S1/(n-m1))
            #Get p-value 
            p = stats.f.sf(f_score,dfd = n-m1, dfn = m1-m0)
            candidate_performance[feature] = p 
        #print(candidate_performance)
        #In the end, insert the feature corresponding to the least p-value 
        if any(np.array(list(candidate_performance.values())) > alpha):
            feature_selected = max(candidate_performance,key = lambda x: candidate_performance[x])
            working_list.remove(feature_selected) #Max if backward
            print(f'Feature {feature_selected} removed from the model')
            features_temp.remove(feature_selected)
            k = features.index(feature_selected)
            A_next = Sweep_k(A_next,k = k ,dk = A[k,k])
            S1 = A_next[z,z]
            m1 = len(working_list)
        else:
            return working_list 

def Best_sub_set(X,y):
    "Return the best selected feature using the least AIC as selection creteria"
    features = list(X.columns)
    n = X.shape[0]
    def AIC(SSR,n,p):
        from math import log
        return n*log(SSR) + 2*p  
    if 'Intercept' not in X:
        X.insert(0,'Intercept',1)
        #features = ['Intercept'] + features
    from itertools import combinations
    Model = []
    for r in range(len(features)):
        r += 1 
        Model += ([(list(np.array(i)+1)) for i in combinations(range(len(features)),r)])
    Model = [(0,)]+[[0] + candidate for candidate in Model]
    X['y'] = y 
    A = np.array(X.T@X)
    candidate_dict = dict()
    for candidate in Model: 
        A_temp = SWEEP(A,candidate,RETURN_INV=1)
        sigma = A_temp[-1,-1]/n
        candidate_dict[tuple(candidate)] = AIC(sigma,n,len(candidate))
    #print(candidate_dict)
    out = min(candidate_dict,key = lambda comb: candidate_dict[comb]) 
    print(candidate_dict[out])
    return [features[idx-1] for idx in out if idx != 0]