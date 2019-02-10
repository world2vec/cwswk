import argparse, os, sys, inspect, logging, time, traceback
import tensorflow as tf

from models import *

import util

gl = globals()


def parse_model_name(model_names):
    return model_names.replace(' ','').split(',')


def create_model(name, seed=None):
    name = name.upper()
    cls = name.split('_')[0]
    if args.suffix is not None:
        name = name + '_' + args.suffix
    cfg = {}
    if seed is not None:
        cfg['seed'] = seed
    for k, v in args.__dict__.items():
        if v is not None:
            cfg[k] = v
    if args.dataset == 'as':
        cfg['min_cnt'] = 2
    model = gl[cls](name = name, cfg = cfg)
    #if args.debug and model.cfg.emb_file is not None:
    #    model.cfg.emb_file = 'debug_' + model.cfg.emb_file


    return model

def pred_model():
    names = parse_model_name(args.model_names)
    for name in names:
        model = create_model(name)
        model.restore(global_step = model.cfg.global_step)
        preds = []
        if args.is_test:
            suffix = 'test'
        else:
            suffix = 'val'
        x, y = util.load_data(args.dataset, suffix, args.debug) 
        x, y = model.process_data(x, y, is_train = False)
        preds = model.predict(x)
        score = util.score(preds, y)
        logging.info('score is %s', score)
        if args.save_model:
            model._save_predict(preds)
        del model
        tf.reset_default_graph()

def train_model(model_names = None):
    if model_names is None:
        model_names = args.model_names

    names = parse_model_name(model_names)


    
    if args.debug:
        dT, yT = util.load_data(args.dataset, 'val', args.debug);
    else:
        dT, yT = util.load_data(args.dataset, 'train', args.debug);
    dV, yV = util.load_data(args.dataset, 'val', args.debug);


    for i, name in enumerate(names):
        logging.info('will train for model: %s', name)
        model = create_model(name)
        if args.restore:
            model.restore(global_step = model.cfg.global_step)
        try:
            xT, yT = model.process_data(dT, yT, is_train = True)
            xV, yV = model.process_data(dV, yV, is_train = False)
            model.fit(xT, yT, xV, yV, args.save_model)
        except Exception as e:
            traceback.print_exc()
            logging.error('*****************************************************************error when fit %sth  model %s:%s', i,name, e)
        finally:
            time.sleep(1)
            logging.info('sleeped 1 second for %sth model  %s',i,name)
            del model
            gc.collect()
            tf.reset_default_graph()

parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
parser.add_argument("-d", "--debug", action = "store_true", help="debug")
parser.add_argument("-save", "--save_model", action = "store_true", help="save")
parser.add_argument("-m", "--model_name", help="model name")
parser.add_argument("-ms", "--model_names", help="model names")
parser.add_argument("-restore", "--restore", action = "store_true", help="save")
parser.add_argument("-s", "--global_step", help="restore epoch")
parser.add_argument("-suf", "--suffix", help="suffex model name")
parser.add_argument("-es", "--es", type=float, help="early stop value")
parser.add_argument("-ed", "--emb_dim", type=int, help="emb dim")
parser.add_argument("-lr", "--lr", type=float, help="early stop value")
parser.add_argument("-bs", "--batch_size", type=int, help="batch_size")
parser.add_argument("-dp", "--dropout", type=float, help="early stop value")
parser.add_argument("-mc", "--mc", type=int, help="min_cnt")
parser.add_argument("-ep", "--epochs", type=int, help="epochs")
parser.add_argument("-ds", "--dataset", help="dataset")
parser.add_argument("-test", "--is_test", action = "store_true", help="run test")

global args
args = parser.parse_args()

if __name__ == '__main__':
    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')

    if args.model_name in gl:
        if inspect.isfunction(gl[args.model_name]):
            gl[args.model_name]()
        else:
            train_model(args.model_name)
    else:
        logging.error('unknown model %s', args.model_name)

                                             

