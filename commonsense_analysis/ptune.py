import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds

class PTuneModel(torch.nn.Module):

    def __init__(self, args, model, tokenizer, template=(3,3,3)):
        super().__init__()
        self.args = args
        self.device = args.device

        self.tokenizer = tokenizer

        self.model = model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embeddings = self.model.get_input_embeddings()

        # set allowed vocab set
        # self.vocab = self.tokenizer.get_vocab()
        # self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))

        # if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
        #     template = (template[0], template[1], 0)
        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args)
        self.prompt_encoder = self.prompt_encoder.to(self.device)
        self.end_token = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        # For using handcraft prompts
        # if self.args.use_original_template:
        #     return raw_embeds

        blocked_indices = (queries == self.pseudo_token_id).nonzero(as_tuple=False).reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def forward(self, batch):
        input_ids = batch["input_ids"]
        bs, l = input_ids.size()

        # get embedded input
        inputs_embeds = self.embed_input(input_ids)

        output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'],
                                return_dict=batch['return_dict'])

        loss, logits = output.loss, output.logits
        # print(logits.shape)
        # print(logits)

        pred_ids = torch.argsort(logits, dim=2, descending=True) # 按顺序排列的id
        pred_rank = torch.argsort(pred_ids, dim=2) # 每个id的rank
        pred_rank = pred_rank.view(bs*l, -1)
        labels = batch['labels'].view(-1) # 每个位置的正确id
        # print(torch.sum((labels != -100)))
        # print(labels)
        # label_mask = (labels != -100).nonzero(as_tuple=False) # 那个位置要算
        # pred_rank = torch.gather(pred_rank, 1, label_mask).view(-1).float()
        # print(pred_ids.shape)
        # print(pred_ids)
        # print(pred_rank.shape)
        # print(pred_rank)
        # exit()

        pred_rank_ = []
        for i, l in enumerate(labels):
            if l != -100:
                pred_rank_.append(pred_rank[i][l])
        pred_rank = torch.tensor(pred_rank_).float().to(self.args.device)


        rank1 = torch.sum((pred_rank == 0))
        rank10 = torch.sum((pred_rank < 10))
        rank100 = torch.sum((pred_rank < 100))


        return loss, (pred_rank, rank1, rank10, rank100)