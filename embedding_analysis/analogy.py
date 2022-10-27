import sys
import os
sys.path.append(os.getcwd())

from model.oscar import OscarModel
from model.voken import AutoVokenModel
from model.univl import UniVLModel
from model.bert import BertModel
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import faiss
import sklearn.model_selection
import csv
from itertools import product

MODEL_CLASSES = {
    "oscar": (AutoConfig, OscarModel, AutoTokenizer),
    "vinvl": (AutoConfig, OscarModel, AutoTokenizer),
    "voken": (AutoConfig, AutoVokenModel, AutoTokenizer),
    "univl": (AutoConfig, UniVLModel, AutoTokenizer),
    "bert": (AutoConfig, BertModel, AutoTokenizer)
}


def collate_fun(batch_data):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    for data in batch_data:
        input_ids.append(torch.LongTensor(data["input_ids"]))
        token_type_ids.append(torch.LongTensor(data["token_type_ids"]))
        attention_mask.append(torch.LongTensor(data["attention_mask"]))

    input_ids = pad_sequence(input_ids, batch_first=True).to(args.device)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True).to(args.device)
    attention_mask = pad_sequence(attention_mask, batch_first=True).to(args.device)

    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask,
           "output_hidden_states": True, 'return_dict': True}
    
class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer

        self.data = []
        for d in data:
            d = self.tokenizer.encode_plus(d)
            for k, v in d.items():
                d[k] = v[1:-1]
            self.data.append(d)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def bats_dataset(args):
    datas = {}
    # types_cnt = {}
    words = []
    n = 0
    types = os.listdir(args.data_path)
    for t in types:
        if '.json' in t:
            types.remove(t)
    for t in types:
        datas[t] = {}
        # types_cnt[t] = 0
        files = os.listdir(os.path.join(args.data_path, t))
        for f_ in files:
            tid = f_.split('.')[0]
            datas[t][tid] = []
            with open(os.path.join(args.data_path, t, f_), "r") as f:
                for line in f.readlines():
                    if line.strip() == '':
                        continue
                    line = line.split('\t')
                    w1 = line[0].strip()
                    w2 = line[1].strip().split('/')
                    w2 = [w for w in w2 if w != '']
                    datas[t][tid].append((w1, w2))

                    n += 1

                    for w in [w1] + w2:
                        if w not in words:
                            words.append(w)

    print('Loaded {} datas and {} words in Total:'.format(n, len(words)))
    for base_t,v in datas.items():
        print(base_t)
        for t, vv in v.items():
            print(t, len(vv))
    return datas, words

def google_dataset(args):
    datas = []
    types_cnt = {}
    words = []
    with open(args.data_path, 'r') as f:
        csv_reader = csv.reader(f)
        _ = next(csv_reader)  # 读取第一行每一列的标题
        for row in tqdm(csv_reader):  # 将csv 文件中的数据保存到birth_data中
            data_type = row[1]
            w1, w2, w3 = row[2], row[3], row[4]
            targets = row[5].split('/')
            for w in [w1, w2, w3]:
                if w not in words:
                    words.append(w)

            for w in targets:
                if w not in words:
                    words.append(w)
            
            if data_type not in types_cnt:
                types_cnt[data_type] = 0
            types_cnt[data_type] += 1
            datas.append((data_type, w1, w2, w3, targets))

    print('Loaded {} datas and {} words in Total:'.format(len(datas), len(words)))
    print(types_cnt)
    return datas, words, types_cnt

def normed(v):
    # if self.normalize:
    #     return v
    # else:
    return v / np.linalg.norm(v)

def find_analogy(embs, words, index, a_, b_, c_):
    a = embs[words.index(a_)]
    b = embs[words.index(b_)]
    c = embs[words.index(c_)]
    d = c - a + b

    d = normed(d)

    _, idxes = index.search(d.reshape((1,-1)), 15)
    analogy = [words[i] for i in idxes[0] if words[i] not in [a_, b_, c_]]
    analogy = analogy[:10]
    
    return analogy

def main(args):
    _, Model, Tokenizer = MODEL_CLASSES[args.model_type]
    tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)

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

    if args.data_type == 'google':
        datas, words, types_cnt = google_dataset(args)

        my_dataset = MyDataset(words, tokenizer)
        word_loader  = DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fun)

        embs = []
        model.eval()
        print('get embedding...')
        with torch.no_grad():
            for batch in tqdm(word_loader):
                output = model(**batch).hidden_states
                output = torch.mean(output[0], dim=1).squeeze()
                # output = model.embeddings.word_embeddings(batch['input_ids'])
                # output = torch.mean(output, dim=1).squeeze()
                embs.append(output.cpu())

        embs = torch.stack(embs).numpy()
        norm = np.linalg.norm(embs, axis=-1).reshape(-1,1)
        embs = embs/norm

        index = faiss.IndexFlatIP(embs.shape[1])
        if args.gpu_mode:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, args.gpu_index, index)
        index.add(embs)

        res = {}
        for t in list(types_cnt.keys()):
            res[t] = {
                'h1': 0,
                'h5': 0,
                'h10': 0,
            }
        res['all'] = {
                'h1': 0,
                'h5': 0,
                'h10': 0,
            }

        for d in tqdm(datas):
            data_type, w1, w2, w3, targets = d
            analogy = find_analogy(embs, words, index, w1, w2, w3)

            for k in [1,5,10]:
                if len(list(set(analogy[:k]).intersection(set(targets)))) != 0:
                    res[data_type][f'h{k}'] += 1
                    res['all'][f'h{k}'] += 1

        for t,n in types_cnt.items():
            for k,v in res[t].items():
                res[t][k] = v/n

        for k,v in res['all'].items():
            res['all'][k] = v/len(datas)

    if args.data_type == 'bats':
        datas, words = bats_dataset(args)

        my_dataset = MyDataset(words, tokenizer)
        word_loader  = DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fun)

        embs = []
        model.eval()
        print('get embedding...')
        with torch.no_grad():
            for batch in tqdm(word_loader):
                output = model(**batch).hidden_states
                output = torch.mean(output[0], dim=1).squeeze()
                embs.append(output.cpu())

        embs = torch.stack(embs).numpy()
        norm = np.linalg.norm(embs, axis=-1).reshape(-1,1)
        embs = embs/norm

        index = faiss.IndexFlatIP(embs.shape[1])
        if args.gpu_mode:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, args.gpu_index, index)
        index.add(embs)

        size_cv_test = 1
        res = {}
        for base_t, all_pairs in datas.items():
            res[base_t] = {}
            for t in list(all_pairs.keys()):
                res[base_t][t] = {
                    'h1': 0,
                    'h5': 0,
                    'h10': 0,
                }
            res[base_t]['all'] = {
                    'h1': 0,
                    'h5': 0,
                    'h10': 0,
                }

            all_cnt = 0
            for t, pairs in all_pairs.items():
                kfold = sklearn.model_selection.KFold(n_splits=len(pairs) // size_cv_test)
                cnt_splits = kfold.get_n_splits(pairs)
                loo = kfold.split(pairs)

                cnt = 0
                for train, test in tqdm(loo, desc=f'{base_t}-{t}'):
                    pairs_test = [pairs[i] for i in test]
                    pairs_train = [pairs[i] for i in train]
                    # p_train = [x for x in p_train if not is_pair_missing(x)]
                    # details += self.solver.do_test_on_pairs(p_train, p_test)
                    for p_train, p_test in product(pairs_train, pairs_test):
                        cnt += 1
                        all_cnt += 1

                        analogy = find_analogy(embs, words, index, p_train[0], p_train[1][0], p_test[0])
                        targets = p_test[1]

                        for k in [1,5,10]:
                            if len(list(set(analogy[:k]).intersection(set(targets)))) != 0:
                                res[base_t][t][f'h{k}'] += 1
                                res[base_t]['all'][f'h{k}'] += 1

                for k,v in res[base_t][t].items():
                    res[base_t][t][k] = v/cnt

            for k,v in res[base_t]['all'].items():
                res[base_t]['all'][k] = v/all_cnt



    print(f'save to {args.save_path}')
    os.makedirs('/'.join(args.save_path.split('/')[:-1]), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(res, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", type=str, default='bert')
    parser.add_argument("--model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--data_path", type=str, default='data/google-analogies.csv')
    parser.add_argument("--data_type", type=str, default='google')
    parser.add_argument("--save_path", type=str, default='linguistic_analysis/res/google_analogy/bert.json')
    parser.add_argument("--gpu_mode", action="store_true")
    parser.add_argument("--gpu_index", type=int, default=0)
    args = parser.parse_args()

    main(args)