import numpy as np
import math
from audioop import reverse

'''
    Distance matching algorithm
    To avoid plug and play users, 
        adopting policy in eval.py using this implementation and following the paper to accquire the best results
'''

def Levenshtein(s,t):
    """Return Levenshtein Distance of 2 strings s and t"""
    """
    s, t: input strings
    """
    inf=1e9
    n=len(s)
    m=len(t)
    d=np.full((n + 1 , m + 1), inf)
    for i in range(n+1):
        d[i,0]=i
    for j in range(m+1):
        d[0,j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            if(s[i-1]==t[j-1]):
                d[i,j]=d[i-1,j-1]
            else:
                d[i,j]=min(d[i,j],min(d[i - 1,j] + 1, min(d[i,j - 1] + 1, d[i - 1,j - 1] + 1)))
    return int(d[n,m])

def topk_bestmatch(ground_truth, input_string, k):
    """Return numpy array indices of topk strings in ground_truth set"""
    """
    ground_truth: array of strings
    input_string: input string to calc Levenshtein Distance
    k           : top k of indices
    """
    result=np.array([])
    for components in ground_truth:
        result=np.append(result,np.array([-Levenshtein(components,input_string)]))
    ind = np.argpartition(result, -k)[-k:]
    ind = ind[np.argsort(result[ind])]
    ind = ind[::-1]
    topk = ind
    return topk, -result[topk]