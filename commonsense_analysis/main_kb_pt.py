import os
import sys

sys.path.append(os.getcwd())

import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data_pt import Atomic2020Data, PTDataset
from kb_utils import atomic2020_relations

from ptune import PTuneModel
from tensorboardX import SummaryWriter

import numpy as np
import json

import argparse

from model.oscar import OscarModelForMaskedLM
from model.voken import VokenModelForMaskedLM
from model.univl import UniVLModelForMaskedLM
from model.bert import BertForMaskedLM
from transformers import AutoTokenizer, AutoConfig

MODEL_CLASSES = {
    "oscar": (AutoConfig, OscarModelForMaskedLM, AutoTokenizer),
    "vinvl": (AutoConfig, OscarModelForMaskedLM, AutoTokenizer),
    "voken": (AutoConfig, VokenModelForMaskedLM, AutoTokenizer),
    "univl": (AutoConfig, UniVLModelForMaskedLM, AutoTokenizer),
    "bert": (AutoConfig, BertForMaskedLM, AutoTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_mode:
        torch.cuda.manual_seed_all(args.seed)

class Trainer(object):    
    def __init__(self, args):
        set_seed(args)

        _, Model, Tokenizer = MODEL_CLASSES[args.model_type]
        self.tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)

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

        template = (3,0,3) if args.use_temp else (3,3,3)
        self.model = PTuneModel(args, model, self.tokenizer, template)
        print("Done.")

        splits = ["train", "dev", "test"]
        print("Loading Data")
        self.datas = {}
        for split in splits:
            data_path = os.path.join(args.data_dir, '{}.json'.format(split))
            self.datas[split] = Atomic2020Data(data_path, args.max_data_for_each_rel)
        print("Done.")

        self.logger = SummaryWriter(os.path.join(args.save_dir, 'runs'))
        print("Logging Tensorboard Files at: {}".format(self.logger.logdir))
        self.args = args

        params = [{'params': self.model.prompt_encoder.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.decay_rate)


    def collate_fn(self, batch_data):
        input_ids = []
        attention_mask = []
        label_ids = []
        for data in batch_data:
            input_ids.append(torch.LongTensor(data["input_ids"]))
            attention_mask.append(torch.LongTensor(data["attention_mask"]))
            label_ids.append(torch.LongTensor(data["label_ids"]))
        
        input_ids = pad_sequence(input_ids, batch_first=True).to(self.args.device)
        attention_mask = pad_sequence(attention_mask, batch_first=True).to(self.args.device)
        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100).to(self.args.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "output_attentions": False,
            "return_dict": True
            }

    def get_save_path(self):
        return os.path.join(self.args.save_dir, 'models')

    def get_checkpoint(self, epoch_idx, dev_score, test_score):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, 
                                                            round(dev_score, 4),
                                                            round(test_score, 4))
        ckpt_name = 'best.ckpt'
        return {'embedding': self.model.prompt_encoder.state_dict(),
                'dev_score': dev_score,
                'test_score': test_score,
                'ckpt_name': ckpt_name,
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        print('Saving model checkpoint to {}'.format(path))
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, os.path.join(path, ckpt_name))
        print("# Checkpoint {} saved.".format(ckpt_name))

    def evaluate(self, epoch_idx, evaluate_type='dev'):
        print('# Evaluating {}...'.format(evaluate_type))
        rels = atomic2020_relations
        if self.args.debug:
            rels = rels[:3]

        self.model.eval()
        dataset = self.datas['test'] if evaluate_type == 'test' else self.datas['dev']
        
        with torch.no_grad():
            res = {}
            mean_loss = []
            all_mean_rank = []
            for rel in rels:
                dataset_ = PTDataset(self.args, dataset, self.tokenizer, rel=rel)
                # loader = DataLoader(dataset_, batch_size=self.args.batch_size,   
                #                     shuffle=False, num_workers=0, collate_fn=self.collate_fn)
                loader = DataLoader(dataset_, batch_size=self.args.batch_size,   
                                    shuffle=False, num_workers=0)
                losses, pred_ranks = [], []
                rank1s, rank10s, rank100s = 0, 0, 0
                cnts = 0
                pbar = tqdm(desc="Calcualting {} score: ".format(rel), total=len(loader))
                if len(loader) == 0:
                    pbar.close()
                    continue
                for batch in loader:
                    batch = self.batch_data(batch)
                    cnt = torch.sum(batch['labels'] != -100, dim=-1)
                    cnts += torch.sum(batch['labels'] != -100)
                    loss, (pred_rank, rank1, rank10, rank100) = self.model(batch)
                    loss = torch.sum(loss.view(batch['labels'].shape), dim=-1)/cnt

                    losses.append(loss)
                    mean_loss.append(loss)
                    pbar.update(batch['input_ids'].size(0))

                    pred_ranks.append(pred_rank)
                    all_mean_rank.append(pred_rank)
                    rank1s += rank1
                    rank10s += rank10
                    rank100s += rank100

                pbar.close()

                losses = torch.cat(losses)
                mlm_score = torch.mean(losses).item()
                std = (torch.std(losses)/np.sqrt(losses.size(0))).item()
                rank1 = rank1s / cnts
                rank10 = rank10s / cnts
                rank100 = rank100s / cnts
                mean_rank = torch.mean(torch.cat(pred_ranks))

                # perplexity = round(perplexity, 2)
                mlm_score = round(mlm_score, 4)
                std = round(std, 4)

                # print('{}: mlm_score: {:4f}, std: {:4f}'.format(rel, mlm_score, std))
                res[rel] = {
                    'mlm_score': mlm_score,
                    'std': std,
                    'mean_rank': mean_rank.item(),
                    'rank@1': rank1.item(),
                    'rank@10': rank10.item(),
                    'rank@100': rank100.item(),
                }
                self.logger.add_scalar("{}/{}/mlm_score".format(evaluate_type, rel), mlm_score, epoch_idx)
                self.logger.add_scalar("{}/{}/mean_rank".format(evaluate_type, rel), mean_rank.item(), epoch_idx)
                self.logger.add_scalar("{}/{}/rank@1".format(evaluate_type, rel), rank1.item(), epoch_idx)
                self.logger.add_scalar("{}/{}/rank@10".format(evaluate_type, rel), rank10.item(), epoch_idx)
                self.logger.add_scalar("{}/{}/rank@100".format(evaluate_type, rel), rank100.item(), epoch_idx)

            all_mean_rank = torch.mean(torch.cat(all_mean_rank)).item()
            mean_loss = torch.cat(mean_loss)
            mean_mlm_score = torch.mean(mean_loss).item()
            std = (torch.std(mean_loss)/np.sqrt(mean_loss.size(0))).item()
            for rel in res.keys():
                print("# {} MLM Score: {:4f}, {:4f}".format(rel, res[rel]['mlm_score'], res[rel]['std']))
            print("# Total MLM Score: {:4f}, {:4f}".format(mean_mlm_score, std))
            self.logger.add_scalar("{}/mlm_score".format(evaluate_type), mean_mlm_score, epoch_idx)

            res['all'] = {
                'mean_rank': all_mean_rank,
                'mlm_score': mean_mlm_score
            }

        return mean_mlm_score, res

    def save_res(self, dev_res, test_res=None):
        path = os.path.join(self.args.save_dir, 'res')
        print('Saving generated results to {}'.format(path))
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'dev.json'), "w") as f:
            json.dump(dev_res, f)
        if test_res is not None:
            with open(os.path.join(path, 'test.json'), "w") as f:
                json.dump(test_res, f)

    def batch_data(self, entry):
        input_ids = entry['input_ids'].to(self.args.device)
        attention_mask = entry['attention_mask'].to(self.args.device)
        label_ids = entry['label_ids'].to(self.args.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "output_attentions": False,
            "return_dict": True
            }
        

    def train(self):
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O2',
                                          verbosity=0)

        print("Training")
        counts = 0
        early_stop = 0
        best_ckpt = None

        print("Creating Train Loader")
        dataset_ = PTDataset(self.args, self.datas['train'], self.tokenizer, rel=self.args.rels)
        # train_loader = DataLoader(dataset_, batch_size=self.args.batch_size,   
        #                     shuffle=True, num_workers=0, collate_fn=self.collate_fn)
        train_loader = DataLoader(dataset_, batch_size=self.args.batch_size,   
                            shuffle=True, num_workers=0)

        for epoch_idx in range(self.args.epoch):
            if epoch_idx > -1:
                dev_mlm_score, dev_res = self.evaluate(epoch_idx, 'dev')

                if epoch_idx == 0:
                    test_mlm_score, test_res = self.evaluate(epoch_idx, 'test')
                    best_dev = dev_mlm_score

                if epoch_idx > 0 and (dev_mlm_score <= best_dev) or self.args.only_evaluate:
                    test_mlm_score, test_res = self.evaluate(epoch_idx, 'test')

                    best_ckpt = self.get_checkpoint(epoch_idx, dev_mlm_score, test_mlm_score)
                    early_stop = 0
                    best_dev = dev_mlm_score

                    self.save_res(dev_res, test_res)
                    
                    print('# Best Dev MLM SCORE: {}'.format(dev_mlm_score))
                    print('# Best Test MLM SCORE: {}'.format(test_mlm_score))
                    
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        self.save(best_ckpt)
                        print("Early stopping at epoch {}.".format(epoch_idx))
                        return best_ckpt
            if self.args.only_evaluate:
                break

            # run training
            tot_loss = []
            
            for batch in tqdm(train_loader, desc="Training {} EPOCH: ".format(epoch_idx)):
                batch = self.batch_data(batch)
                counts += 1
                self.model.train()
                loss, (pred_rank, rank1, rank10, rank100) = self.model(batch)

                cnt = torch.sum(batch['labels'] != -100, dim=-1)
                loss = torch.sum(loss.view(batch['labels'].shape), dim=-1)/cnt

                mean_rank = torch.mean(pred_rank)

                tot_loss.append(loss)
                loss = torch.mean(loss)

                if self.args.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if args.fp16:
                    total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1.0)
                else:
                    total_norm =torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.logger.add_scalar("train/loss", loss.item(), counts)
                self.logger.add_scalar("train/mean_rank", mean_rank.item(), counts)
                self.logger.add_scalar("train/rank@1", rank1.item(), counts)
                self.logger.add_scalar("train/rank@10", rank10.item(), counts)
                self.logger.add_scalar("train/rank@100", rank100.item(), counts)

            self.my_lr_scheduler.step()

            tot_loss = torch.cat(tot_loss)
            tot_loss = torch.mean(tot_loss)

            print("# total loss: {}".format(tot_loss))
        
        dev_mlm_score, dev_res = self.evaluate(epoch_idx, 'dev')
        if dev_mlm_score <= best_dev:
            test_mlm_score, test_res = self.evaluate(epoch_idx, 'test')
            best_ckpt = self.get_checkpoint(epoch_idx, dev_mlm_score, test_mlm_score)
            best_dev = dev_mlm_score

            self.save_res(dev_res, test_res)

            print('# Best Dev MLM SCORE: {}'.format(dev_mlm_score))
            print('# Best Test MLM SCORE: {}'.format(test_mlm_score))

        self.save(best_ckpt)
        self.logger.close()

        return best_ckpt

def main(args):
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "univl", "oscar", "vinvl", "voken", "unimo"])
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='data/atomic2020')
    parser.add_argument("--save_dir", type=str, default='commonsense_analysis/res/atomic2020')
    parser.add_argument("--max_data_for_each_rel", type=int, default=-1)
    parser.add_argument('--pred', type=str, default='h')
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--use_temp", type=bool, default=True)
    

    parser.add_argument("--gpu_mode", action="store_true")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size when running model (default=32).")  
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument('--rels', nargs='+', default=None)

    parser.add_argument("--debug", type=bool, default=False)

    parser.add_argument("--fp16", action="store_true")


    args = parser.parse_args()

    main(args)