# -*- coding: utf-8 -*-

import os
import re
import util
from collections import defaultdict

PKU_TRAIN='original_data/pku_training.utf8'
PKU_TEST='original_data/pku_test_gold.utf8'
MSR_TRAIN='original_data/msr_training.utf8'
MSR_TEST='original_data/msr_test_gold.utf8'
AS_TRAIN='original_data/as_training.utf8'
AS_TEST='original_data/as_test_gold.utf8'
CITYU_TRAIN='original_data/cityu_training.utf8'
CITYU_TEST='original_data/cityu_test_gold.utf8'
OUTPUT_PATH='data'

rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'

def strQ2B(string):
    """全角转半角"""
    bstring = ""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        bstring += chr(inside_code)
    return bstring

def write_lines(lines, fname):
    is_train = False
    if 'train' in fname:
        is_train = True
        
    dataset = os.path.basename(fname).split('_')[0]
    fdict = os.path.join(OUTPUT_PATH, dataset+'_dict.dump')
    idioms = {}
    with open('data/original_data/idioms') as f:
        for l in f:
            idioms[l.strip()] = 1
    
    new_lines = [];
    for l in lines:
        words = strQ2B(l).split()
        if len(words)==0:
            continue
        new_words = []
        for word in words:
            word = re.sub(rNUM,'0',word)
            word = re.sub(rENG,'X',word)
            if word in idioms:
                word = 'I'
            new_words.append(word)
        new_words.append(os.linesep)
        new_lines.append(' '.join(new_words))
    with open(fname, 'w') as f:
        f.writelines(new_lines)

def preprocess(fname):
    fname = os.path.join('data', fname)
    print('process {}'.format(fname))
    dataset = os.path.basename(fname).split('_')[0]
    sents=[]
    with open(fname) as f:
        lines = f.readlines()
    num = len(lines)

    if 'test' in fname:
        fTest = os.path.join(OUTPUT_PATH, dataset+'_test.txt')
        write_lines(lines, fTest)
    else:
        train_num = int(num*0.9)
        train_lines = lines[0:train_num]
        val_lines = lines[train_num:]
        fT = os.path.join(OUTPUT_PATH, dataset+'_train.txt')
        fV = os.path.join(OUTPUT_PATH, dataset+'_val.txt')
        write_lines(train_lines, fT)
        write_lines(val_lines, fV)
        
if __name__ == '__main__':
    print('start preprocess')
    for dataset in ['pku', 'msr', 'as', 'cityu']:
        fname = os.path.join('original_data', dataset+'_training.utf8')
        preprocess(fname)
        fname = os.path.join('original_data', dataset+'_test_gold.utf8')
        preprocess(fname)




 
    
