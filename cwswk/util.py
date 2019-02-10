import os, pickle, logging
import itertools


def score(preds, tags):
    preds = list(itertools.chain.from_iterable(preds))
    tags = list(itertools.chain.from_iterable(tags))
    assert len(preds) == len(tags)
    pred_num = preds.count(0) + preds.count(1)
    tag_num = tags.count(0) + tags.count(1)
    start = 0; cor_num = 0
    for i in range(len(tags)):
        if tags[i] == 0 or tags[i] == 3:
            flag = True
            for j in range(start, i+1):
                if tags[j] != preds[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i + 1
    P = cor_num / pred_num
    R = cor_num / tag_num
    F = 2*P*R/(P+R)
    return P, R, F

def ngram(chars, ng = (2,5)):
    N = len(chars)
    pads = ['[PAD]']*(ng[1]-ng[0]+1)
    chars = pads+chars+pads
    ngrams = []
    for i in range(len(pads), N+len(pads)):
        ngram = []
        ngram.append(chars[i])
        for j in range(ng[0], ng[1]+1):
            ngram.append(''.join(chars[i-j+1:i+1]))
            ngram.append(''.join(chars[i:i+j]))
        ngrams.append(ngram)
    return ngrams

def ngram_all(chars, ng = 2, window_size = 2):
    N = len(chars)
    pads = ['[PAD]']*(window_size)
    chars = pads+chars+pads
    ngrams = []
    for i in range(window_size, N + window_size):
        ngram = []
        for j in range(i-window_size, i+window_size +1):
            for k in range(ng):
                if (j+k+1)<=(i+window_size+1):
                    ngram.append(''.join(chars[j:j+k+1]))
        ngrams.append(ngram)
    return ngrams

def dump(obj, fname):
    fr = open(fname,'wb')
    pickle.dump(obj,fr)
    fr.close()


def load_exists(fname):
    if os.path.exists(fname):
            print('will load obj from file:' + fname)
            fr = open(fname,'rb')
            obj = pickle.load(fr)
            fr.close()
            return obj

def word2tags(word):
    tags = []
    num = len(word)
    if num == 1:
        tags.append(0)
    elif num >= 2:
        tags.append(1)
        for i in range(num - 2):
            tags.append(2)
        tags.append(3)
    return tags



def words2tags(words):
    tags = []
    for word in words:
        tags.extend(word2tags(word))
    return tags

def load_data(dataset, suffix, debug = False):
    fname = os.path.join('../data', dataset+ '_' + suffix + '.txt')
    logging.info('will load data from file:%s', fname)
    with open(fname) as f:
        lines = f.readlines()
    if debug:
        lines = lines[0:100]
    lines = [l.strip() for l in lines]
    sents = []; tags = []
    for line in lines:
        words = line.split()
        sent = ''.join(words)
        tag = words2tags(words)
        sents.append(sent); tags.append(tag)
    return sents, tags

def parse_arg(args_ns, cfg):
    args = vars(args_ns)
    for key in args:
        if args[key] is not None:
            cfg[key] = args[key]
    

    
