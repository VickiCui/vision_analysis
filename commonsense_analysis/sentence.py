import os
import sys

sys.path.append(os.getcwd())

import random

import torch
import torch.nn as nn
# torch.multiprocessing.set_start_method("spawn")
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json

from utils import metrics

from model.oscar import OscarModel
from model.voken import AutoVokenModel
from model.univl import UniVLModel
from transformers import AutoTokenizer, AutoConfig, AutoModel

from tqdm import tqdm

import faiss
import argparse

def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.gpu_mode:
        torch.cuda.manual_seed_all(config.seed)

MODEL_CLASSES = {
    "oscar": (AutoConfig, OscarModel, AutoTokenizer),
    "vinvl": (AutoConfig, OscarModel, AutoTokenizer),
    "voken": (AutoConfig, AutoVokenModel, AutoTokenizer),
    "univl": (AutoConfig, UniVLModel, AutoTokenizer),
    "bert": (AutoConfig, AutoModel, AutoTokenizer)
}

types = {
    "domains": ["living", "nonliving"], 
    "feature_types": ["visual perceptual", "other perceptual", "taxonomic", "encyclopaedic", "functional"]
}
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class CSLBData(object):
    def __init__(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.concepts = data["concepts"]
            self.features = data["features"]
            self.domains = data["domains"]
            self.feature_types = data["feature_types"]
            self.datas = data["datas"]
        print("{} concepts and {} features in total".format(len(self.concepts), len(self.features)))

class ConceptDataset(Dataset):
    def __init__(self, cslb_data, tokenizer, template=None):
        self.sent = cslb_data.concepts
        if template is not None:
            self.sent = [template.replace("<REP>", s) for s in self.sent]
        self.tokenizer = tokenizer

        self.dataset = []
        for c in self.sent:
            data = self.tokenizer.encode_plus(c)
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class FeatureDataset(Dataset):
    def __init__(self, cslb_data, tokenizer, template=None):
        self.sent = cslb_data.features
        if template is not None:
            self.sent = [template.replace("<REP>", s) for s in self.sent]
        self.tokenizer = tokenizer

        self.dataset = []
        for c in self.sent:
            data = self.tokenizer.encode_plus(c)
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
        
def main(args):
    set_seed(args)
    cslb_data = CSLBData(args.data_path)
    _, Model, Tokenizer = MODEL_CLASSES[args.model_type]
    tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)
    concapt_dataset = ConceptDataset(cslb_data, tokenizer, "something is <REP>")
    feature_dataset = FeatureDataset(cslb_data, tokenizer, "something <REP>")

    print("Building Model from {} ...".format(args.model_name_or_path))
    model = Model.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
    )
    model.eval()

    # Push to GPU
    if args.gpu_mode:
        # args.device = torch.device("cuda", args.gpu_index)
        torch.cuda.set_device(args.gpu_index)
        args.device = args.gpu_index
        print("Pushing to GPU: {}".format(args.device))
        model.cuda(args.device)
    else:
        args.device = 'cpu'

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

        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}

    concept_loader  = DataLoader(concapt_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fun)
    feature_loader  = DataLoader(feature_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fun)
    
    if args.norm:
        norm_layer = LayerNorm(768)
        if args.gpu_mode:
            norm_layer.cuda(args.device)

    # print("Calculating concept representations...")
    # concept_cls = []
    # concept_avg = []
    concept_word = []
    with torch.no_grad():
        for batch in tqdm(concept_loader, desc="Calculating concept representations"):
            output = model(**batch)
            sequence_output = output[0]
            if args.norm:
                sequence_output = norm_layer(sequence_output)
            # cls_output = sequence_output[:, 0]
            word_output = sequence_output[:, 1]
            # avg_output = torch.sum(sequence_output, dim=1)
            # avg_output /= torch.sum(batch['attention_mask'], dim=1)


            # hidden_states = outputs.hidden_states
            # attentions = outputs.attentions[0]
            # caption_hidden_states.append(hidden_states)
            # caption_attentions.append(attentions)

            # concept_cls.append(cls_output)
            # concept_avg.append(avg_output)
            concept_word.append(word_output)

        # concept_cls = torch.cat(concept_cls, dim=0)
        # concept_avg = torch.cat(concept_avg, dim=0)
        concept_word = torch.cat(concept_word, dim=0)

    # print("Calculating feature representations...")
    # feature_cls = []
    # feature_avg = []
    feature_word = []
    with torch.no_grad():
        for batch in tqdm(feature_loader, desc="Calculating feature representations"):
            output = model(**batch)
            sequence_output = output[0]
            if args.norm:
                sequence_output = norm_layer(sequence_output)
            # cls_output = sequence_output[:, 0]
            word_output = sequence_output[:, 1]
            # avg_output = torch.mean(sequence_output, dim=1)

            # feature_cls.append(cls_output)
            # feature_avg.append(avg_output)
            feature_word.append(word_output)

        # feature_cls = torch.cat(feature_cls, dim=0)
        # feature_avg = torch.cat(feature_avg, dim=0)
        feature_word = torch.cat(feature_word, dim=0)
        norm = torch.norm(feature_word, p=2, dim=-1).view(-1,1)
        feature_word = feature_word/norm
        dim = feature_word.shape[-1]

    # a2b_pattern = {
    #     "concept to feature cls": (concept_cls, feature_cls),
    #     "concept to feature word": (concept_word, feature_word),
    #     "concept to feature avg": (concept_avg, feature_avg),
    #     "feature to concept cls": (feature_cls, concept_cls),
    #     "feature to concept word": (feature_word, concept_word),
    #     "feature to concept avg": (feature_avg, concept_avg),
    # }
    a2b_pattern = {
        "concept to feature word": (concept_word, feature_word),
        "feature to concept word": (feature_word, concept_word),
    }
    conditions = ['all']
    # for d in cslb_data.domains:
    #     conditions.append('domain-{}'.format(d))
    for f in cslb_data.feature_types:
        conditions.append('feature_types-{}'.format(f))

    all_res = {}
    # all_res['similarity'] = {}
    # all_res['rand_similarity'] = {}
    all_res['retrieval_score'] = {}
    for condition in conditions:
        all_res['retrieval_score'][condition] = {}


    for pattern_name, (a, b) in a2b_pattern.items():
        # # concept to feature cosin similarity
        # print("============================={}============================".format(pattern_name))
        # print("************************ Calculating cosin similarity ************************")
        # similarities = []
        # for data in tqdm(cslb_data.datas, desc=pattern_name):
        #     domain, feature_type, concept_idx, feature_idx = data
        #     if 'concept to feature' in pattern_name:
        #         query_idx = concept_idx
        #         key_idx = feature_idx
        #     else:
        #         query_idx = feature_idx
        #         key_idx = concept_idx

        #     query_emb = a[query_idx]
        #     key_emb = b[key_idx]

        #     similarity = torch.cosine_similarity(query_emb, key_emb, dim=0)
        #     similarities.append(similarity.cpu().item())

        # mean_similarities = np.mean(similarities)

        # rand_similarities = []
        # for i in tqdm(range(a.size(0)), desc='rand similarity'):
        #     query_emb = a[i]
        #     for j in range(b.size(0)):
        #         key_emb = b[j]
        #         similarity = torch.cosine_similarity(query_emb, key_emb, dim=0)
        #         rand_similarities.append(similarity.cpu().item())

        # mean_rand_similarities = np.mean(rand_similarities)

        # print("mean_similarities: {}".format(round(mean_similarities, 6)))
        # print("mean_rand_similarities: {}".format(round(mean_rand_similarities)))
        # all_res['similarity'][pattern_name] = mean_similarities
        # all_res['rand_similarity'][pattern_name] = mean_rand_similarities

        # rank with faiss
        print("************************ Calculating retrieval score ************************")
        for condition in conditions:
            query2keys = {}
            query_num, key_num = 0,0
            for data in tqdm(cslb_data.datas, desc=pattern_name):
                domain, feature_type, concept_idx, feature_idx = data
                if 'domain' in condition and domain not in condition:
                    continue
                if 'feature_type' in condition and feature_type not in condition:
                    continue

                if 'concept to feature' in pattern_name:
                    query_idx = concept_idx
                    key_idx = feature_idx
                else:
                    query_idx = feature_idx
                    key_idx = concept_idx

                if query_idx not in query2keys:
                    query_num += 1
                    query2keys[query_idx] = []
                query2keys[query_idx].append(key_idx)
                key_num += 1

            index = faiss.IndexFlatIP(dim)
            if args.gpu_mode:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(b.cpu().numpy())

            scores = {
                "HIT@1": 0,
                "HIT@5": 0,
                "HIT@10": 0,
                "MAP@10": 0,
                "MAP@100": 0,
                "MRR@10": 0,
                "MRR@100": 0,
            }

            for query_idx, gt in query2keys.items():
                _, idxes = index.search(a[query_idx].cpu().numpy().reshape(1,-1), 100)
                pred = idxes[0]

                if args.random:
                    all_inedx = range(feature_word.shape[0]) if 'concept to feature' in pattern_name else range(concept_word.shape[0])
                    pred = random.sample(all_inedx, 100)

                for metrics_map in list(scores.keys()):
                    s = metrics(gt, pred, metrics_map)
                    scores[metrics_map] += s
            for metrics_map in list(scores.keys()):
                if "HIT" in metrics_map:
                    scores[metrics_map] /= key_num
                else:
                    scores[metrics_map] /= query_num

            print("{} {} retrieval score:".format(condition, pattern_name))
            str = ''
            for k,v in scores.items():
                str += "{}: {};  ".format(k, round(v, 4)*100)
            print(str)
            all_res['retrieval_score'][condition][pattern_name] = scores
    
    os.makedirs('/'.join(args.save_path.split('/')[:-1]), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(all_res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/CSLB_Property_Norms_V1.1/all.json")
    parser.add_argument("--save_path", type=str, default="./representation_analysis/res/sentence.json")
    parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "univl", "oscar", "vinvl", "voken", "unimo"])
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--gpu_mode", action="store_true")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--random", action="store_true")

    args = parser.parse_args()

    main(args)