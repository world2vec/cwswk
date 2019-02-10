import tensorflow as tf
import numpy as np, os, time, logging, json
import multiprocessing as mp
import util
from functools import partial
from copy import deepcopy

initializer = tf.contrib.layers.xavier_initializer(uniform=False)
dense = partial(tf.layers.dense, kernel_initializer = initializer)

class CFG(object):
    def __init__(self):
        self.seed = 8763
        self.lr = 1e-2
        self.batch_size = 128
        self.lr_decay_rate = 1.0
        self.emb_dim = 64
        self.es = 0.01
        self.epochs = 10
        self.min_cnt = 0
        self.global_step = None
        self.emb_file = None
        self.save_model = False
        self.data_dir = '../data'
        self.debug = True
        self.qsize = 100
        self.dataset = ''
        self.batch_wait = 30

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)

    def save(self, fpath):
        with open(fpath, 'w') as f:
            json.dump(self.__dict__, f)

    def load(self, fpath):
        with open(fpath) as f:
            cfg = json.load(f)
        self.update(cfg)
    def copy(self):
        return deepcopy(self)



class Model():
    cfg = CFG()
    def __init__(self, name = None, cfg = {}):
        self.name = name
        if self.name is None:
            self.name = self.__class__.__name__
        self.cfg = self.cfg.copy()
        self.cfg.update(cfg)

        self._rs = np.random.RandomState(self.cfg.seed)

        self._graph = None
        self.sess = None
        self.summary_nodes = []
        self.loss_nodes = []
        self.validate_loss_nodes = []

        self._model_dir = self.gen_fname('', self.cfg.dataset)
        if self.cfg.save_model:
            logging.info('model dir is %s', self._model_dir)
            if not os.path.exists(self._model_dir):
                os.makedirs(self._model_dir)


    @property
    def graph(self):
        return self._graph

    @property
    def loss(self):
        return self._loss
    
    def create_graph(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self._sess = tf.Session(config = config)
            tf.set_random_seed(self.cfg.seed)
            self._create_graph()

    def _create_graph(self):
        self._add_plh()


    def _add_plh(self):
        raise NotImplementedError

    def _add_emb(self, inputs, size, dim = None, name='emb', embedding = None, trainable=True, dropout = None):
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        if embedding is None:
            embed_var = tf.get_variable(name + '_W', [size, dim], tf.float32, initializer = initializer, trainable = trainable)
        else:
            embed_var = tf.get_variable(name + '_W', initializer = embedding, trainable = trainable)
        embeded = tf.nn.embedding_lookup(embed_var, inputs, name = name)
        return embeded, embed_var

    def _create_cell(self, enc_dims, cell_cls):
        cells = []
        for layer in enc_dims:
            cells.append(cell_cls(layer))
        mult_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        return mult_cell

    def _add_rnn(self, inputs, enc_dims, cell_cls, bi = True, name = 'rnn'):
        initializer = tf.orthogonal_initializer()
        with tf.variable_scope(name, initializer = initializer):
            cell = self._create_cell(enc_dims, cell_cls)
            if bi:
                cell_bw = self._create_cell(enc_dims, cell_cls)
                outs, stats = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, sequence_length = self._input_seq_len_plh,  swap_memory = True, dtype=tf.float32)
            else:
                outs, stats= tf.nn.dynamic_rnn(cell, inputs, sequence_length = self._input_seq_len_plh,  swap_memory = True)
        return outs, stats


    def _add_feed(self, feed, batch, name): 
        feed[self._graph.get_tensor_by_name(name + '_plh:0')] =  batch[name]

    def get_feed(self, batch):
        raise NotImplementedError

    def run(self, sess, batch, nodes):
        feed = self.get_feed(batch)
        outputs = sess.run(nodes, feed)
        return outputs


    def save(self, sess, model_name = 'model', global_step = None):
        model_dir = self._model_dir
        fpath = os.path.join(model_dir, 'cfg.json')
        self.cfg.save(fpath)

        with sess.graph.as_default():
            saver = tf.train.Saver()
        save_path = saver.save(sess, os.path.join(model_dir, model_name), global_step = global_step)
        logging.info("Model saved to file:{}".format(save_path))

    def restore(self, model_name = "model", global_step = None):
        model_dir = self._model_dir
        fpath = os.path.join(model_dir, 'cfg.json')
        self.cfg.load(fpath)

        if self._graph is None:
            self.create_graph()
        sess = self._sess
        with sess.graph.as_default():
            saver = tf.train.Saver()
            model_path = os.path.join(model_dir, model_name)
            if global_step is None:
                ckpt = tf.train.get_checkpoint_state(model_dir)
                model_path = ckpt.model_checkpoint_path
            else:
                model_path = model_path + "-{}".format(global_step)
            global_step = model_path.split('-')[1].split('.')[0]
            saver.restore(sess, model_path)
            logging.info("Model restored from file:{}".format(model_path))

        return global_step

    def gen_fname(self, postfix, *paras):
        if self.cfg.debug:
            name = os.path.join(self.cfg.data_dir, 'debug_' + self.name +  '_' + postfix, *paras)
        else:
            name = os.path.join(self.cfg.data_dir, self.name  + '_' + postfix, *paras)
        return name

    def _get_batch(self, x, y = None, shuffle = True):
        raise NotImplementedError
        

    def get_batch(self, x, y = None, shuffle = True):
        queue = mp.Queue(self.cfg.qsize)
        def _gen_batch(x, y, shuffle):
            for i, batch in enumerate(self._get_batch(x, y, shuffle)):
                queue.put(batch, block=True)
            queue.put(None)
            queue.close()

        thread = mp.Process(target=_gen_batch, args=[x, y, shuffle])
        thread.daemon = True
        thread.start()
        logging.info('get batch wait time %s seconds', self.cfg.batch_wait)
        while True:
            batch = queue.get(True, self.cfg.batch_wait)
            if batch is None:break;
            yield batch


    def _save_predict(self, pred, suffix=''):
        fpath = self.gen_fname('', 'pred' + suffix + '.dump')
        util.dump(pred, fpath)

    def seq(self, sents, w2i, max_len = 3000):
        pad = w2i['[PAD]']
        seqs = []; seq_lens = []
        for sent in sents:
            inds = [w2i[w] if w in w2i else w2i['[UNK]'] for w in sent]
            seq_len = len(inds)
            inds += [pad]*(max_len  - len(inds))
            seqs.append(inds)
            seq_lens.append(seq_len)
        return np.array(seqs), np.array(seq_len)

    def _get_lr(self, itr):
        return self.cfg.lr
    def _fit_epoch(self, x, y, xV = None, yV = None, epoch = None):
        losses = []; train_outstr = ""; val_losses = []; validate_outstr = ""
        num = len(x)//self.cfg.batch_size + 1
        for node in self.loss_nodes:
            losses.append([])
            train_outstr += node.name + ":{},"

        for node in self.validate_loss_nodes:
            val_losses.append([])
            validate_outstr += node.name + ":{},"
        itr = 0
        for i, batch in enumerate(self._get_batch(x, y, shuffle=True)):
            #batch['lr'] = self.cfg.lr
            batch['lr'] = self._get_lr(itr)
            itr += 1
            outs = self.run(self._sess, batch, self.loss_nodes + [self._global_step, self._train_op])
            #outs = self.run(self._sess, batch, [self.loss, self._global_step, self._summary_op, self._train_op])
            for j, node in enumerate(self.loss_nodes):
                losses[j].append(outs[j])
            global_step = outs[j+1]; summary = outs[j+2]
            #self._summary_writer.add_summary(summary, global_step)
            #self._summary_writer.flush()
            #sys.stdout.write(('\r' + outstr).format(*map(np.mean, losses)))
            #sys.stdout.flush()
            if (i+1) % 30 == 0:
                logging.info('name:%s,global step:%s,train loss is:%s, totally %s batchs', self.name,global_step, train_outstr.format(*map(np.mean, losses)), i)
        logging.info('name:%s,global step:%s,train loss is:%s, totally %s batchs', self.name,global_step, train_outstr.format(*map(np.mean, losses)), i)
        loss = np.sum(list(map(np.mean, losses)))
        #if not no_val:
        if 1==1:
            for i, batch in enumerate(self._get_batch(xV, yV, shuffle=False)):
                outs = self.run(self._sess, batch, self.validate_loss_nodes + [self._global_step])
                for j, node in enumerate(self.validate_loss_nodes):
                    val_losses[j].append(outs[j])
                global_step = outs[j+1]
            val_loss = np.sum(list(map(np.mean, val_losses)))
            logging.info(("epoch:{}, val loss is " + validate_outstr).format(epoch, *map(np.mean, val_losses)))
        else:
            val_loss = None
        #pred = self.predict(x)
        #logging.info("train score is %s", self.score(x,y))
        return loss, val_loss

    def fit(self, x, y, xV = None, yV = None, save = True, restore = False):
        if self._graph is None:
            self.create_graph()
        if restore:
            self.restore(self.cfg.global_step)

        with self._graph.as_default():
            best_val_loss = np.inf
            epochs = self.cfg.epochs
            for i in range(epochs):
                loss, val_loss = self._fit_epoch(x, y, xV, yV, epoch=i)
                score = self.score(xV, yV)

                if save:
                    self.save(self._sess, global_step = i)

                gain = (best_val_loss - val_loss) / val_loss
                if gain < self.cfg.es:
                    logging.info("gain(%s) < es(%s), train done", gain, self.cfg.es)
                    break
                else:
                    logging.info("gain(%s) >= es(%s), train continue", gain, self.cfg.es)
                    if best_val_loss > val_loss:
                        best_val_loss = val_loss
            #if save:
            #    pred = self.predict(xV)
            #    self._save_predict(pred, '')
