import gc, logging, os
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.contrib import crf

import util
from basic_models import Model, CFG, dense


class CWS(Model):
    cfg = Model.cfg.copy()
    cfg.emb_file = os.path.join(cfg.data_dir, 'w2v_100.txt')
    cfg.emb_dim = 100
    cfg.dropout = 0.8
    cfg.cws_enc_dims = [64]
    cfg.cws_num_classes = 4
    cfg.window_size = 2
    cfg.max_grad = 5
    cfg.l2 = 1e-4
    _bi = True
    _cell_cls = tf.nn.rnn_cell.LSTMCell
    _fake = {'[PAD]':0, '[SOS]': 1, '[EOS]':2, '[UNK]':3 }

    def _add_plh(self):
        self._lr_plh = tf.placeholder(tf.float32, name = "lr_plh" )
        num_ng = self.cfg.window_size*4 + 1
        self._input_plh = tf.placeholder(tf.int32, [None, None, num_ng], name = "seqs_plh" )
        self._input_seq_len_plh = tf.placeholder(tf.int32,[None], name = 'seqs_len_plh')
        self._dropout_plh = tf.placeholder(tf.float32,name = 'dropout_plh')
        self._cws_tag_plh = tf.placeholder(tf.int32, [None, None], name = "cws_tags_plh" )

    def _add_loss(self):
        self._loss_crf = tf.reduce_mean(-self._crf_log_likelihood, name = 'crf_loss')
        self._l2_loss = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if 'bias' not in var.name:
                self._l2_loss.append(self.cfg.l2*tf.nn.l2_loss(var))

        self._l2_loss = tf.add_n(self._l2_loss, name='l2_loss')

        self._loss = tf.add_n([self._loss_crf, self._l2_loss], name='loss')
        self.loss_nodes = [self._loss_crf, self._l2_loss]
        self.validate_loss_nodes = [self._loss_crf]

    def _post_loss(self):
        self._var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self._train_op, self._var_list, self._gradients = self._add_train_op(self._loss, self._var_list)
        self._var_init = tf.global_variables_initializer()
        self._model = self
        self._sess.run(self._var_init)


    def _create_graph(self):
        self._add_plh()
        self._add_cws()
        self._add_loss()
        self._post_loss()

    def _load_emb(self, fname):
        logging.info('will use emb file :%s', fname)
        num = len(self.cfg.w2i)+1
        embeddings = np.random.uniform(-0.5,0.5,[num, self.cfg.emb_dim]).astype(np.float32)
        with open(fname, errors = 'ignore') as f:
            for l in f:
                items = l.strip().split()
                if items[0] in self.cfg.w2i:
                    embeddings[self.cfg.w2i[items[0]]] = np.array(items[1:]).astype(np.float32)
        return embeddings
            
    def _add_cws_emb(self):
        if self.cfg.emb_file:
            embeddings = self._load_emb(self.cfg.emb_file)
        else:
            embeddings = None
        self._cws_embeded, self._emb_w = self._add_emb(self._input_plh, self.cfg.emb_size, self.cfg.emb_dim, embedding = embeddings)
        shape = tf.shape(self._cws_embeded)
        dim = self._cws_embeded.get_shape().as_list()[-1]
        self._cws_embeded = tf.reshape(self._cws_embeded, [shape[0],shape[1],(self.cfg.window_size*4+1)*dim])
        if self.cfg.dropout is not None:
            self._cws_embeded = tf.nn.dropout(self._cws_embeded, self._dropout_plh)

    def _add_cws(self):
        with tf.variable_scope("cws", reuse = tf.AUTO_REUSE):
            self._add_cws_emb()
            self._cws_outs, self._cws_states = self._add_rnn(self._cws_embeded, self.cfg.cws_enc_dims, self._cell_cls, self._bi)
            if self._bi:
                self._cws_outs = tf.concat(self._cws_outs,-1)
            if self.cfg.dropout is not None:
                self._cws_outs = tf.nn.dropout(self._cws_outs, self._dropout_plh)
            self._cws_logits = dense(self._cws_outs, self.cfg.cws_num_classes)
            self._crf_log_likelihood, self._crf_transition_params = crf.crf_log_likelihood(self._cws_logits[:,0:], self._cws_tag_plh[:,0:], self._input_seq_len_plh)
    def _cnt2dic(self, cnt, min_cnt):
        w2i = self._fake.copy(); ind = len(w2i)
        rare_words = []
        for key , v in sorted(cnt.items()):
            if v >= min_cnt:
                w2i[key] = ind
                ind += 1
            else:
                rare_words.append(key)
        return w2i, rare_words

    def gen_dict(self, sents, min_cnt):
        cnt = Counter()
        pads = ['[PAD]']*self.cfg.window_size
        for sent in sents:
            chars = pads + list(sent) + pads
            ngrams = []
            for i in range(len(chars)-len(pads)):
                for j in range(self.cfg.window_size):
                    ngrams.append(''.join(chars[i:i+j+1]))
            cnt.update(ngrams)
                
        self.cfg.w2c = dict(cnt)
        w2i, rare_words = self._cnt2dic(cnt, min_cnt)
        self.cfg.w2i = w2i
        self.cfg.wmc = dict(cnt.most_common(int(len(cnt)*0.2)))
        self.cfg.emb_size = len(w2i)

        word_len = [(k,len(k)) for k, v in w2i.items() if v>=4]
        word_len = sorted(word_len, key=lambda x:x[-1], reverse=True)

        logging.info('dict num is:%s, word len is:%s, rare_words num:%s, %s', len(w2i), word_len[0:10], len(rare_words), rare_words[0:10])

    def seq(self, sents, w2i, max_len = 3000):
        pads = ['[PAD]'] * self.cfg.window_size
        seqs = []; seq_lens = []
        for sent in sents:
            chars = list(sent)
            seq_len = len(chars)
            ngrams = util.ngram_all(chars, 2, self.cfg.window_size)
                
            inds = [[w2i[w] if w in w2i else w2i['[UNK]'] for w in ngram] for ngram in ngrams]
            seqs.append(inds)
            seq_lens.append(seq_len)
        return seqs, seq_lens

    def process_data(self, x, y, is_train = False):
        if is_train:
            self.gen_dict(x, self.cfg.min_cnt)
        seqs, seqs_len = self.seq(x, self.cfg.w2i)
        x = {'sents': np.array(x),
             'seqs':np.array(seqs),
             'seqs_len': np.array(seqs_len),
            }
        return x, np.array(y)
    

    def _get_batch(self, x, y = None, shuffle = True):
        batch_size = self.cfg.batch_size
        num = len(x['sents'])
        inds = np.arange(num)
        if shuffle:
            self._rs.shuffle(inds)
        num_batch = (num + batch_size -1)//batch_size
        for i in range(num_batch):
            batch = {}
            batch_inds = inds[batch_size*i:min(batch_size*(i+1), num)]
            batch['batch_inds'] = batch_inds
            for key in x:
                batch[key] = x[key][batch_inds]

            max_len = np.max(batch['seqs_len'])
            batch_num = len(batch_inds)
            seqs = np.zeros([batch_num, max_len, self.cfg.window_size*4+1], np.int32)
            for i in range(batch_num):
                seq = batch['seqs'][i]
                seqs[i, 0:len(seq)] = np.array(seq)
            batch['seqs'] = seqs

            if y is not None:
                tags = np.zeros([batch_num, max_len], np.int32)
                for i, tag in enumerate(y[batch_inds]):
                    tags[i,0:len(tag)] = tag
                batch['cws_tags'] = tags
            yield batch


    def get_feed(self, batch):
        feed = {}
        self._add_feed(feed, batch, 'seqs')
        self._add_feed(feed, batch, 'seqs_len')
        if 'cws_tags' in batch:
            self._add_feed(feed, batch, 'cws_tags')
        if 'lr' in batch:
            self._add_feed(feed, batch, 'lr')
            if self.cfg.dropout is not None:
                feed[self._dropout_plh] = self.cfg.dropout
        else:
            if self.cfg.dropout is not None:
                feed[self._dropout_plh] = 1.0

        return feed

    def predict(self, x):
        preds = []; sents = []; tags = []
        for i, batch in enumerate(self._get_batch(x, None, shuffle=False)):
            lengths = batch['seqs_len']; sents = batch['sents']
            cws_logits, transition_params = self.run(self._sess, batch, [self._cws_logits, self._crf_transition_params])
            for i, length in enumerate(lengths):
                logits = cws_logits[i]
                sequence, _ = crf.viterbi_decode(logits, transition_params)
                sequence = sequence[0:length]
                preds.append(sequence)
        return preds


    def score(self, x, y):
        preds = self.predict(x)
        precision, recall, score = util.score(preds, y)
        logging.info('precision %s, recall %s, score is %s', precision, recall, score)
        return score

    def _add_train_op(self, loss, var_list = None, name='adam'):
        self._global_step = tf.Variable(0, trainable=False, name = 'global_step')
        opt = tf.train.AdamOptimizer(learning_rate=self._lr_plh, name=name)

        max_grad = self.cfg.max_grad
        if var_list is None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = opt.compute_gradients(loss, var_list)
        clipped = [(grad if grad is None else tf.clip_by_norm(grad, max_grad), var) for grad, var in gradients]
        opt_op = opt.apply_gradients(clipped, global_step=self._global_step, name = name)
        return opt_op, var_list, gradients

class CWSWK(CWS):
    cfg = CWS.cfg.copy()
    cfg.ext_fdict = 'jieba_dict.txt'
    cfg.ngram = 5
    cfg.use_bert = True
    cfg.bert_batch_size = 32

    
    def _get_ext_dict(self):
        self._dict = {}
        with open(os.path.join(self.cfg.data_dir, self.cfg.ext_fdict)) as f:
            for i, l in enumerate(f):
                items = l.strip().split()
                self._dict[items[0]] = np.log(1+float(items[1]))
    def create_graph(self):
        super(CWSWK, self).create_graph()
        if self.cfg.use_bert:
            self._add_bert()

    def _create_graph(self):
        super(CWSWK, self)._create_graph()
    def _add_bert(self):
        import bert_model
        self._bert = bert_model.BERT()
        self._bert.create_graph()
    def _add_plh(self):
        super(CWSWK, self)._add_plh()
        self._ngd_feas_plh = tf.placeholder(tf.float32,[None, None, 2*(self.cfg.ngram-1)],name = 'ngd_feas_plh')
        self._bert_feas_plh = tf.placeholder(tf.float32,[None, None, 768],name = 'bert_feas_plh')
        
    def _add_cws_emb(self):
        if self.cfg.use_bert:
            self._cws_embeded = self._bert_feas_plh
        else:
            super(CWSWK, self)._add_cws_emb()
        if self.cfg.ext_fdict:
            self._cws_embeded = tf.concat([self._cws_embeded, self._ngd_feas_plh],-1)
        
    def get_feed(self, batch):
        feed = super(CWSWK, self).get_feed(batch)
        if self.cfg.ext_fdict:
            self._add_feed(feed, batch, 'ngd_feas')
        if self.cfg.use_bert:
            self._add_feed(feed, batch, 'bert_feas')
        return feed
    def _get_batch(self, x, y = None, shuffle = True):
        if self.cfg.ext_fdict:
            self._get_ext_dict()
        for batch in super(CWSWK, self)._get_batch(x, y, shuffle):
            if self.cfg.ext_fdict:
                ng_feas = np.zeros([len(batch['sents']), batch['seqs'].shape[1], 2*(self.cfg.ngram -1)], np.float32)
                for i, sent in enumerate(batch['sents']):
                    chars = list(sent)
                    ngs = util.ngram(chars, (2, self.cfg.ngram))
                    ng_fea = []
                    for j, ng in enumerate(ngs):
                        ng = ng[1:]#remove the first 1-gram character itself
                        ng_fea.append([self._dict[x] if x in self._dict else 0 for x in ng]) 
                    ng_feas[i,0:len(ng_fea)] = np.array(ng_fea)
                batch['ngd_feas'] = ng_feas
            if self.cfg.use_bert:
                max_len = np.max(batch['seqs_len'])
                feas = self._bert.extract_feas(batch, self.cfg.bert_batch_size)
                bert_feas = np.zeros([len(feas), max_len, 768 ], np.float32)
                for i, fea in enumerate(feas):
                    bert_feas[i, 0:len(fea)] = fea
                batch['bert_feas'] = bert_feas

                
            yield batch


class CWSB(CWSWK):
    cfg = CWSWK.cfg.copy()
    cfg.ext_fdict = None
    cfg.use_bert = True
    cfg.bert_proj_dim = 64
    def _add_cws_emb(self):
        super(CWSWK, self)._add_cws_emb()
        proj = dense(self._bert_feas_plh, self.cfg.bert_proj_dim)
        self._cws_embeded = tf.concat([self._cws_embeded, proj],-1)

class CWSD(CWSWK):
    cfg = CWSWK.cfg.copy()
    cfg.use_bert = False
    cfg.ext_fdict = 'jieba_dict.txt'
    def _add_cws_emb(self):
        super(CWSWK, self)._add_cws_emb()
        self._cws_embeded = tf.concat([self._cws_embeded, self._ngd_feas_plh],-1)

class CWSBD(CWSB):
    cfg = CWSB.cfg.copy()
    cfg.ext_fdict = 'jieba_dict.txt'
    def _add_cws_emb(self):
        super(CWSBD, self)._add_cws_emb()
        self._cws_embeded = tf.concat([self._cws_embeded, self._ngd_feas_plh],-1)
