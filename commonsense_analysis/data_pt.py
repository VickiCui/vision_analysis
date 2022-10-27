import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
import os
import json
import pandas as pd
from toolz.sandbox import unzip
from kb_utils import split_into_words, atomic2020_templates

class ConceptNetData(object):
    def __init__(self, data_path, max_data_for_each_rel=100000):
        self.datas = {}
        with open(data_path, 'r') as f:
            datas_ = json.load(f)

        self.rels = []
        for rel, datas in datas_.items():
            if rel not in split_into_words:
                continue
            self.rels.append(rel)
            datas = datas[:max_data_for_each_rel]
            self.datas[rel] = \
                [(d['head'].lower().strip(), split_into_words[rel].strip(),
                d['tail'].lower().strip()) for d in datas]

class Atomic2020Data(object):
    def __init__(self, data_path, max_data_for_each_rel=-1):
        self.datas = {}
        with open(data_path, 'r') as f:
            datas_ = json.load(f)
        self.rels = []
        for rel, datas in datas_.items():
            if rel not in atomic2020_templates:
                continue
            self.rels.append(rel)
            if max_data_for_each_rel != -1:
                datas = datas[:max_data_for_each_rel]
            if rel == 'isFilledBy':
                self.datas[rel] = \
                    [(d[0].replace('___', '<OBJ>'), None, d[2].replace('___', 'something')) for d in datas]
            else:
                template = atomic2020_templates[rel]
                self.datas[rel] = \
                    [(template, d[0].replace('___', 'something'), d[2].replace('___', 'something')) for d in datas]

        # data = pd.read_csv(data_path, delimiter='\t')
        # data.columns = ['s', 'r', 'o']
        # self.rels = []
        # for i, d in tqdm(data.iterrows()):
        #     try:
        #         s = d['s'].strip()
        #         r = d['r'].strip()
        #         o = d['o'].strip()
        #     except:
        #         continue
        #     if s == 'none' or o == 'none':
        #         continue
        #     if r not in atomic2020_templates:
        #         continue
        #     if r not in self.datas:
        #         self.datas[r] = set()
        #         self.rels.append(r)

        #     self.datas[r].add((s,r,o))
            
        # for rel, v in self.datas.items():
        #     v = list(v)
        #     if max_data_for_each_rel != -1:
        #         v = v[:max_data_for_each_rel]

        #     if rel == 'isFilledBy':
        #         self.datas[rel] = \
        #             [(d[0].replace('___', '<OBJ>'), None, d[2].replace('___', 'something')) for d in v]
        #     else:
        #         template = atomic2020_templates[rel]
        #         self.datas[rel] = \
        #             [(template, d[0].replace('___', 'something'), d[2].replace('___', 'something')) for d in v]


class PTDataset(Dataset):
    def __init__(self, args, atomic2020_data, tokenizer, rel=None):
        super().__init__()
        self.data = []
        self.args = args

        self.tokenizer = tokenizer
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.vocab_encoder = tokenizer
        self.vocab_decoder = None
        self.template = (3,0,3) if args.use_temp else (3,3,3)

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        datas_ = atomic2020_data.datas
        datas = []
        if rel is None:
            for _,v in datas_.items():
                datas.extend(v)
        else:
            if isinstance(rel, str):
                rels = [rel]
            for rel in rels:
                datas.extend(datas_[rel])

        if self.args.debug:
            datas = datas[:3000]

        for d in datas:
            x_h = d[1]
            x_t = d[2]
            temp = d[0]
            example = self.get_query(x_h, x_t, pred=args.pred, temp=temp, use_temp=args.use_temp)
            if example is not None:
                self.data.append({
                    'input_ids': example[0],
                    'attention_mask': example[1],
                    'label_ids': example[2]
                })

        print('Loaded {} data'.format(len(self.data)))
        

    def get_query(self, x_h, x_t, pred='h', temp=None, use_temp=True):
        prompt_tokens = [self.pseudo_token_id]

        if pred == 'h' and x_h is None: # special case for 'isFilledBy' relation
            return None

        if pred == 't':
            label_tokens = self.tokenizer.tokenize(x_t)

            if use_temp:
                mask = ['[MASK]'] * len(label_tokens)
                mask = ' '.join(mask)
                if x_h is None:
                    temp = temp.replace('<OBJ>', mask)
                else:
                    temp = temp.replace('<SUBJ>', x_h).replace('<OBJ>', mask)

                input_ids =  ([self.tokenizer.cls_token_id]
                + prompt_tokens * self.template[0]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(temp))
                + prompt_tokens * self.template[2]
                + [self.tokenizer.sep_token_id])

            else:
                input_ids =  ([self.tokenizer.cls_token_id]
                + prompt_tokens * self.template[0]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_h))
                + prompt_tokens * self.template[1]
                + [self.tokenizer.mask_token_id] * len(label_tokens)
                + prompt_tokens * self.template[2]
                + [self.tokenizer.sep_token_id])

        if pred == 'h':
            label_tokens = self.tokenizer.tokenize(x_h)

            if use_temp:
                mask = ['[MASK]'] * len(label_tokens)
                mask = ' '.join(mask)
                temp = temp.replace('<OBJ>', x_t).replace('<SUBJ>', mask)

                input_ids = ([self.tokenizer.cls_token_id]
                + prompt_tokens * self.template[0]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(temp))
                + prompt_tokens * self.template[2]
                + [self.tokenizer.sep_token_id])

            else:
                input_ids =  ([self.tokenizer.cls_token_id]
                + prompt_tokens * self.template[0]
                + [self.tokenizer.mask_token_id] * len(label_tokens)
                + prompt_tokens * self.template[1]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))
                + prompt_tokens * self.template[2]
                + [self.tokenizer.sep_token_id])

        if len(label_tokens) > 5:
            return None

        if self.tokenizer.unk_token_id in input_ids:
            return None
        if len(input_ids) > 32:
            return None

        # print(self.tokenizer.tokenize(temp))

        l0 = input_ids.index(self.tokenizer.mask_token_id)
        l1 = l0 + len(label_tokens)

        # label_ids = [-100] * len(input_ids)
        # label_ids[l0:l1] = self.tokenizer.convert_tokens_to_ids(label_tokens)
        
        # label_ids = torch.Tensor(label_ids).long()
        # input_ids = torch.Tensor(input_ids).long()
        # attention_mask = torch.ones(input_ids.shape).long()

        len_input = len(input_ids)
        pad_len = 32 - len_input

        label_ids = [-100] * 32
        label_ids[l0:l1] = self.tokenizer.convert_tokens_to_ids(label_tokens)

        attention_mask = [1] * len_input + [0] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        label_ids = torch.Tensor(label_ids).long()
        input_ids = torch.Tensor(input_ids).long()
        attention_mask = torch.Tensor(attention_mask).long()

        # print(x_h)
        # print(temp)
        # print(x_t)
        # print(input_ids)
        # print(attention_mask)
        # print(label_ids)
        # exit()
        
        return (input_ids, attention_mask, label_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

 