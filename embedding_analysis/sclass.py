# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import pickle
import json
import logging
import os
import random
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers.data.metrics import acc_and_f1

from transformers import (
    AdamW,
    PreTrainedModel,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
# from transformers import glue_compute_metrics as compute_metrics
# from transformers import glue_convert_examples_to_features as convert_examples_to_features
# from transformers import glue_output_modes as output_modes
# from transformers import glue_processors as processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

def get_WIKI_PSE(args, path, tokenizer, model, cache_dir):
    datas = []
    all_tags = set()
    model.eval()

    if not args.debug and os.path.exists(cache_dir):
        with open(cache_dir, 'rb') as f:
            data = pickle.load(f)
        return data['datas'], data['all_tags']

    with torch.no_grad():
        # self.embedding = model.embeddings.word_embeddings

        with open(path, 'r') as f:
            csv_reader = csv.reader(f, delimiter ='\t')
            for i, row in tqdm(enumerate(csv_reader)):
                if args.debug and i >= 1000:
                    break

                word = row[0][1:-1].replace('_', ' ').strip()
                if '[UNK]' in tokenizer.tokenize(word):
                    continue
                input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))

                if len(input_ids) > 1 and args.one_token:
                    continue
                else:
                    tags = row[1].split(' ')
                    tags = [r.split('=')[0][1:] for r in tags if int(r.split('=')[1])>=3]
                    if len(tags) == 0:
                        continue
                    all_tags.update(tags)
                    input_ids = torch.Tensor(input_ids).view(1, -1).long().to(args.device)

                    try:
                        inputs_embeds = model(input_ids=input_ids, output_hidden_states = True, return_dict = True).hidden_states[0]
                    except:
                        print(row)
                        continue
                    if not 'rand' in cache_dir:
                        inputs_embeds = torch.mean(inputs_embeds, dim=1).squeeze()
                        inputs_embeds = inputs_embeds.cpu().numpy()
                    else:
                        inputs_embeds = torch.randn(768).numpy()

                    # inputs_embeds = self.embedding(input_ids)
                    # inputs_embeds = torch.mean(inputs_embeds, dim=1).view(-1)

                    datas.append({
                        'word': word,
                        'inputs_embeds': inputs_embeds,
                        'tags': tags,
                    })

    print(len(datas))
    if not args.debug:
        with open(cache_dir, 'wb') as f:
            data = {
                'datas': datas,
                'all_tags': list(all_tags),
            }
            pickle.dump(data, f)

    return datas, list(all_tags)


class SClassDataset(Dataset):
    '''
    {
        "txt1": "The city councilmen refused the demonstrators a permit because", 
        "pron": "they", 
        "txt2": "feared violence.", 
        "trigger": "feared violence", 
        "answer": ["city councilmen", " councilmen"], 
        "correct": 0
    }
    '''
    def __init__(self, args, datas, split, tag):
        self.tag = tag
        n = 5000 if len(datas) >= 30000 else 500
        if not args.debug:
            if split == 'train':
                datas = datas[:-n]
            elif split == 'val':
                datas = datas[-n:]
            else:
                datas = datas

        self.dataset = self.prepare(datas)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def prepare(self, datas):
        dataset = []

        for d in datas:
            label = 1 if self.tag in d['tags'] else 0
            
            data = {
                'input_embedding': d['inputs_embeds'],
                'label': label
            }

            dataset.append(data)

        return dataset


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # return self.transform(self.dataset[idx]['input_embedding']), self.transform(self.dataset[idx]['label'])
        return self.dataset[idx]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        # pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.view(-1,1)
        target = target.view(-1,1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1-pred,pred),dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor. 
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

         # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class MyModel(nn.Module):
    def __init__(self, softmax=True):
        super().__init__()
        self.softmax = softmax
        if self.softmax:
            self.MLP = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2),
            )
        else:
            self.MLP = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )
    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     """ Initialize the weights """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()

    def forward(self, input_embeddings, labels):
        logits = self.MLP(input_embeddings)

        if self.softmax:
            preds = logits.argmax(dim=-1)
            loss_fct = nn.CrossEntropyLoss()
        else:
            preds = (logits >= 0.5).long()
            loss_fct = FocalLoss()

            
        loss = loss_fct(logits, labels.view(-1))

        

        return loss, logits, preds


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(args, name, model):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

def collate_fun(batch_data):
    input_embeddings, labels = [], []
    for data in batch_data:
        input_embeddings.append(torch.FloatTensor(data["input_embedding"]))
        labels.append(data["label"])
    
    input_embeddings = torch.stack(input_embeddings)
    labels = torch.LongTensor(labels)

    return input_embeddings, labels

def train(args, dataset, model, tag):
    """ Train the model """
    train_dataset = SClassDataset(args, dataset, 'train', tag)
    val_dataset = SClassDataset(args, dataset, 'val', tag)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fun, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_warmup_steps = int(t_total * args.warmup_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch Size = %d", args.train_batch_size)

    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )

    best_f1 = 0.
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            model.train()
            
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {"input_embeddings": batch[0], "labels": batch[1]}
            outputs = model(**inputs) # loss, logits, preds
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}

                    results = evaluate(args, val_dataset, model, tag)
                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    #for key, value in logs.items():
                        #tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                    f1 = results['f1']
                    if f1 >= best_f1:
                        best_f1 = f1
                        save_model(args, "best-checkpoint", model)

#             break
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        

        logs = {}
                        
        results = evaluate(args, val_dataset, model, tag)
        for key, value in results.items():
            eval_key = "eval_{}".format(key)
            logs[eval_key] = value

        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
        learning_rate_scalar = scheduler.get_lr()[0]
        logs["learning_rate"] = learning_rate_scalar
        logs["loss"] = loss_scalar
        logging_loss = tr_loss

        print(json.dumps({**logs, **{"step": global_step}}))

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    #for key, value in logs.items():
        #tb_writer.add_scalar(key, value, global_step)
    print(json.dumps({**logs, **{"step": global_step}}))

    f1 = results['f1']
    if f1 >= best_f1:
        best_f1 = f1
        save_model(args, "best-checkpoint", model)

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, tag):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, collate_fn=collate_fun, batch_size=args.train_batch_size)


    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = tag
    eval_output_dir = args.output_dir

    results = {}


    # Eval!
    logger.info("***** Running evaluation {} *****".format(eval_task))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {"input_embeddings": batch[0], "labels": batch[1]}
            outputs = model(**inputs) # loss, logits, preds
            tmp_eval_loss, logits, preds_ = outputs
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1



        if preds is None:
            preds = preds_.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, preds_.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    result = acc_and_f1(preds, out_label_ids)
    results.update(result)
    
#         print(preds, out_label_ids)
    # os.makedirs(os.path.join(eval_output_dir, tag), exist_ok=True)
    # output_eval_file = os.path.join(eval_output_dir, tag, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     # logger.info("***** Eval results {} *****".format(tag))
    #     for key in sorted(result.keys()):
    #         # logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="The input data cache dir. Should contain the .pkl files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        #help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--one_token", action="store_true", help="Set this flag to filter multi-tokens word.",
    )

    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=-1, help="Log every X updates steps.")

    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
    parser.add_argument("--gpu_idx", type=int, default=0, help="random seed for initialization")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--softmax", action="store_true")


    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.gpu_idx == -1:
        device = torch.device("cpu")
        args.n_gpu = 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.gpu_idx)
        device = torch.device("cuda", args.gpu_idx)
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.warning(
        "Process device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
#     model = AutoModelForSequenceClassification(config=config)
    data_model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
    )
    data_model.to(args.device)

    os.makedirs(args.cache_dir, exist_ok=True)
    path = os.path.join(args.data_dir, 'train.tsv')
    cache_dir = os.path.join(args.cache_dir, 'train.pkl')
    data_for_train, all_tags1 = get_WIKI_PSE(args, path, tokenizer, data_model, cache_dir)
    path = os.path.join(args.data_dir,'test.tsv')
    cache_dir = os.path.join(args.cache_dir, 'test.pkl')
    data_for_test, _ = get_WIKI_PSE(args, path, tokenizer, data_model, cache_dir)

    # if not args.debug:
    #     assert len(all_tags1) == len(all_tags2), 'tags in train and test data are not same'
    all_tags = all_tags1

    all_test_results = {}
    all_val_results = {}
    
    for tag in all_tags:
        logger.info(" ****************** %s ******************", tag)
        model = MyModel(args.softmax)
        model.to(args.device)

        # Training
        if args.do_train:
            global_step, tr_loss = train(args, data_for_train, model, tag)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        # Evaluation

        val_results = {}
        if args.do_eval:
            val_dataset = SClassDataset(args, data_for_train, 'val', tag)
            checkpoint = os.path.join(args.output_dir, 'best-checkpoint', 'pytorch_model.bin')

            model = MyModel(args.softmax)
            model.eval()
            if args.gpu_idx != -1:
                ckpt = torch.load(checkpoint)
            else:
                ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt)
            model.to(args.device)
            result = evaluate(args, val_dataset, model, tag)
            val_results.update(result)

        all_val_results[tag] = val_results


        test_results = {}
        if args.do_test:
            test_dataset = SClassDataset(args, data_for_test, 'val', tag)
            checkpoint = os.path.join(args.output_dir, 'best-checkpoint', 'pytorch_model.bin')

            model = MyModel(args.softmax)
            model.eval()
            if args.gpu_idx != -1:
                ckpt = torch.load(checkpoint)
            else:
                ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt)
            model.to(args.device)
            result = evaluate(args, test_dataset, model, tag)
            test_results.update(result)

        all_test_results[tag] = test_results

    mean_f1 = []
    for _, res in all_val_results.items():
        mean_f1.append(res['f1'])
    mean_f1 = np.mean(mean_f1)
    logger.info("val mean_f1 = %s", mean_f1)
    all_val_results['mean_f1'] = mean_f1
    
    mean_f1 = []
    for _, res in all_test_results.items():
        mean_f1.append(res['f1'])
    mean_f1 = np.mean(mean_f1)
    logger.info("test mean_f1 = %s", mean_f1)
    all_test_results['mean_f1'] = mean_f1


    with open(os.path.join(args.output_dir, 'val_results.json'), 'w') as f:
        json.dump(all_val_results, f)
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(all_test_results, f)


if __name__ == "__main__":
    main()