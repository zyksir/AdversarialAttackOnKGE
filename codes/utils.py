import os
import logging
import datetime
import argparse
import pickle
import datetime
import json
import logging
import os
import random
from IPython import embed
import numpy as np
import torch

class InputData(object):
    def __init__(self, entity2id, relation2id, train_triples, valid_triples, test_triples, all_true_triples,
                 fake_triples=None):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        self.all_true_triples = all_true_triples
        self.fake_triples = fake_triples

def get_input_data(args):
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    args.nentity, args.nrelation = len(entity2id), len(relation2id)

    train_triples = read_triple(os.path.join(args.data_path, "train.txt"), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    all_true_triples = train_triples + valid_triples + test_triples
    fake_triples = []
    if args.fake:
        if args.fake == "empty":
            fake_triples = []
        else:
            fake_triples = pickle.load(open(os.path.join(args.save_path, "%s.pkl" % args.fake), "rb"))
        train_triples += fake_triples
        test_triples = pickle.load(open(os.path.join(args.data_path, "targetTriples.pkl"), "rb"))

    logging.info(args.comments)
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % args.nentity)
    logging.info('#relation: %d' % args.nrelation)
    logging.info('#train: %d\t#valid: %d\t#test: %d' % (len(train_triples), len(valid_triples), len(test_triples)))

    return InputData(entity2id=entity2id,
                     relation2id=relation2id,
                     train_triples=train_triples,
                     valid_triples=valid_triples,
                     test_triples=test_triples,
                     all_true_triples=all_true_triples,
                     fake_triples=fake_triples)

def get_parser():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument("--fake", type=str, default=None, help="fake data used")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--data_path', type=str, default=None, help="dataset we used")
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    # about model saving, usually we should NOT save models to save space
    parser.add_argument('--no_save', action='store_true', help='do not save models')
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--comments', default="\n", type=str)
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=2000, type=int, help='train log every xx steps')
    parser.add_argument('--classify_steps', default=5000, type=int, help='regiven weight log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    return parser

def parse_args(args=None):
    parser = get_parser()
    return parser.parse_args(args)

def checkArgsValidation(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('you must set save_path for log and model file saving')

def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def set_logger(args, filename="train"):
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    today = datetime.datetime.now()
    log_file = os.path.join(args.save_path or args.init_checkpoint, '%s-%d-%d.log' % (filename, today.month, today.day))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples
