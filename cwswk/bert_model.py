from bert import modeling
from bert import tokenization as tkz


import tensorflow as tf
from models import CWS
import numpy as np,logging, sys, os

class BERT(CWS):
    cfg = CWS.cfg.copy()
    cfg.bert_config_file =  os.path.join(cfg.data_dir,"bert/chinese_L-12_H-768_A-12/bert_config.json")
    cfg.vocab_file =  os.path.join(cfg.data_dir, "bert/chinese_L-12_H-768_A-12/vocab.txt")
    cfg.init_checkpoint = os.path.join(cfg.data_dir, "bert/chinese_L-12_H-768_A-12/bert_model.ckpt")
    cfg.batch_size = 16

    tokenizer = tkz.BasicTokenizer()
    def get_feed(self, batch):
        feed = {self._input_plh: batch['seqs'],
                self._input_seq_len_plh: batch['seqs_len'],
                }
        return feed

    def _add_plh(self):
        self._input_plh = tf.placeholder(tf.int32, [None, None], name = "input_plh" )
        self._input_seq_len_plh = tf.placeholder(tf.int32,[None], name = 'input_seq_len_plh')
    def _create_graph(self):
        self._add_plh()
        self.vocab = tkz.load_vocab(self.cfg.vocab_file)
        mask = tf.cast(tf.sequence_mask(self._input_seq_len_plh,tf.shape(self._input_plh)[1]), tf.int32)
        bert_config = modeling.BertConfig.from_json_file(self.cfg.bert_config_file)
        self._bert_model = modeling.BertModel(config = bert_config, is_training = False, input_ids = self._input_plh, input_mask = mask,
                               use_one_hot_embeddings = False)
        self._feas = self._bert_model.get_sequence_output()
        self._var_init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self._sess = tf.Session(config=config, graph = self._graph)
        self.init_var()
    def init_var(self):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        #(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.cfg.init_checkpoint)
        #tf.train.init_from_checkpoint(self.cfg.init_checkpoint, assignment_map)
        with self._sess.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._sess, self.cfg.init_checkpoint)
    def extract_feas(self, ori_batch, batch_size = 16):
        num = len(ori_batch['sents'])
        inds = np.arange(num)
        num_batch = (num + batch_size -1)//batch_size
        feas = []; ori_sent_id = 0
        for i in range(num_batch):
            batch = {}
            batch_inds = inds[batch_size*i:min(batch_size*(i+1), num)]
            batch['batch_inds'] = batch_inds
            batch['sent_ids'] = batch_inds
            batch['sents'] = ori_batch['sents'][batch_inds]
            batch = self.bert_batch(batch)
            fea = self.run(self._sess, batch, self._feas)
            pre_sent_id = -1; comb_fea = []
            for f, ind, sent_id in zip(fea, batch['ori_to_tok_inds'], batch['sent_ids']):
                if sent_id != pre_sent_id and len(comb_fea)>0:
                    feas.append(np.concatenate(comb_fea))
                    comb_fea = []
                comb_fea.append(f[ind])
                pre_sent_id = sent_id
            if len(comb_fea)>0:
                feas.append(np.concatenate(comb_fea))
                    
        return feas
        
    def seq(self, chars):    
        seq = []
        for char in chars:
            if char in self.vocab:
                seq.append(self.vocab[char])
            else:
                seq.append(self.vocab['[UNK]'])
        return seq
    def bert_batch(self,batch, bert_max_len = 512):
        sents = batch['sents']; sent_ids = batch['sent_ids']
        ori_to_tok_inds = []
        seqs = []; max_len = 0; seqs_len = []
        new_sent_ids = []
        for sent, sent_id in zip(sents, sent_ids):
            offset = 0
            ori_to_tok_ind = []
            chars = [char for char in sent if char !=' ']
            new_chars = ['[CLS]']
            ind = 1#0 for CLS
            for i, char in enumerate(chars):
                ori_to_tok_ind.append(ind)
                for new_char in self.tokenizer.tokenize(char):
                    new_chars.append(new_char)
                    ind += 1
                    if ind>=(bert_max_len-2):
                        new_chars.append('[SEP]')
                        ori_to_tok_inds.append(ori_to_tok_ind)
                        ori_to_tok_ind = []
                        if len(new_chars)>max_len:
                            max_len = len(new_chars)
                        seqs.append(self.seq(new_chars))
                        seqs_len.append(len(new_chars))
                        new_sent_ids.append(sent_id)
                        new_chars = ['[CLS]']
                        ind = 1
            if len(new_chars)>1:
                new_chars.append('[SEP]')
                ori_to_tok_inds.append(ori_to_tok_ind)
                ori_to_tok_ind = []
                if len(new_chars)>max_len:
                    max_len = len(new_chars)
                seqs.append(self.seq(new_chars))
                seqs_len.append(len(new_chars))
                new_sent_ids.append(sent_id)
        for i, seq in enumerate(seqs):
            seqs[i] = seq + [0] * (max_len - len(seq))
        batch['seqs'] = np.array(seqs)
        batch['ori_to_tok_inds'] = ori_to_tok_inds
        batch['seqs_len'] = seqs_len
        batch['sent_ids'] = new_sent_ids
        return batch
