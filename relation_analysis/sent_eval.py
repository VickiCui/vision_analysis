# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
sys.path.append(os.getcwd())

import torch
import random

import pickle
import logging
import argparse

from model.oscar import OscarModel
from model.voken import AutoVokenModel
from model.univl import UniVLModel
from transformers import AutoTokenizer, AutoConfig, AutoModel

# Set PATHs
PATH_SENTEVAL = 'data/SentEval'
PATH_TO_DATA = 'data/SentEval/data'

# V = 1 # version of InferSent

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


MODEL_CLASSES = {
    "oscar": (AutoConfig, OscarModel, AutoTokenizer),
    "vinvl": (AutoConfig, OscarModel, AutoTokenizer),
    "voken": (AutoConfig, AutoVokenModel, AutoTokenizer),
    "univl": (AutoConfig, UniVLModel, AutoTokenizer),
    "bert": (AutoConfig, AutoModel, AutoTokenizer)
}

# Set the random seeds
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)

def prepare(params, samples):
    return

def sentence_emb(sent, mask=None, type_sent='avg'):
    # sent: BxLx768
    # mask: BxL
    if type_sent == 'avg':
        sent = sent * mask.unsqueeze(-1)
        sum = torch.sum(mask, -1)
        sentvec = torch.sum(sent, 1) / sum.unsqueeze(-1)
        sentvec = sentvec.cpu().numpy()
    elif type_sent == 'cls':
        sentvec = sent[:, 0].cpu().numpy()
    return sentvec

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    batch = [' '.join(s) for s in batch]

    with torch.no_grad():
        sentence_inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        for k,v in sentence_inputs.items():
            sentence_inputs[k] = v.to(args.device)
        sentence_inputs["return_dict"] = True
        sentence_inputs["output_hidden_states"] = True

        sentence_embedding = params['model'](**sentence_inputs).hidden_states
        sentence_embedding = sentence_embedding[params['layer']]
        embeddings = sentence_emb(sentence_embedding, sentence_inputs['attention_mask'], params['type_sent'])

    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "univl", "oscar", "vinvl", "voken", "unimo"])
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--type_sent", type=str, default='cls', choices=["cls", "avg"], help='cls or avg')
    parser.add_argument("--layer", type=int, default=12, help='0~12, 0: embedding, 1~12: each layer')
    parser.add_argument("--gpu_mode", action="store_true")
    parser.add_argument("--gpu_index", type=int, default=0)
    args = parser.parse_args()

    _, Model, Tokenizer = MODEL_CLASSES[args.model_type]
    tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)
    
    print("Building Model from {} ...".format(args.model_name_or_path))
    model = Model.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
    )

    # Push to GPU
    if args.gpu_mode:
        # args.device = torch.device("cuda", args.gpu_index)
        torch.cuda.set_device(args.gpu_index)
        args.device = args.gpu_index
        print("Pushing to GPU: {}".format(args.device))
        model.cuda(args.device)
    else:
        args.device = 'cpu'

    params_senteval['model'] = model
    params_senteval['tokenizer'] = tokenizer
    params_senteval['type_sent'] = args.type_sent
    params_senteval['layer'] = args.layer
    params_senteval['device'] = args.device

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['Length', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'CoordinationInversion', 'OddManOut']
    results = se.eval(transfer_tasks)
    print(results)

    os.makedirs('/'.join(args.save_path.split('/')[:-1]), exist_ok=True)
    print('Saving results to {}...'.format(args.save_path))
    with open(args.save_path, "wb") as f:
        pickle.dump(results, f)