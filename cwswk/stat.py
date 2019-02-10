import os, sys, numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def stat(dataset):
    fname = os.path.join('data', dataset + '_train.txt')
    with open(fname) as f:
        lines = f.readlines()
    dic = defaultdict(int)
    for l in lines:
        for w in l.strip().split():
            dic[w] += 1
    lens = [len(w) for w in dic]
    hist, bins = np.histogram(lens, bins = [0,3,5,10,100,1000])
    print(hist)

if __name__ == '__main__':
    for dataset in sys.argv[1].split():
        stat(dataset)
    
