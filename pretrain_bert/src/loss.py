from __future__ import print_function

import torch
import torch.nn as nn
from apex import amp

class ListMLE(nn.Module):
    def __init__(self, ignore=-100):
        super(ListMLE, self).__init__()
        self.ignore = ignore

        self.forward = amp.float_function(self.forward_impl)

    def labels_to_list_label(self, labels, class_number):
        """Convert masked lm labels to one hot format,

            Args:
                labels: labels of shape [bsz, length].
                class_number: number of class
            Returns:
                list_labels: labels of shape [bsz*length, n_class]
        """
        device = labels.device
        batch_size, length = labels.size()
        idxs = torch.nonzero(~torch.eq(labels, self.ignore))
        value = labels[idxs[:,0], idxs[:,1]]
        one_hot = torch.ones(batch_size, class_number) * self.ignore
        index = (idxs[:,0], value)
        one_hot_labels = one_hot.index_put(index, torch.ones(value.size())).long()
        one_hot_labels = one_hot_labels.to(device) # B x N

        list_labels = one_hot_labels.unsqueeze(1).repeat(1, length, 1) # B x L x N
        list_labels = list_labels.view(-1, class_number) # B*L x N
        labels = labels.view(-1) # B*L
        idxs = torch.nonzero(~torch.eq(labels, self.ignore))
        value = labels[idxs[:]]
        list_labels[idxs[:], value[:]] = 2

        idxs = torch.nonzero(torch.eq(labels, self.ignore))
        list_labels[idxs[:], :] = self.ignore


        return list_labels

    def forward_impl(self, logits, labels, syn_w=0.1, eps=1e-12, reduction='mean'):
        """[summary]

        Args:
            logit ([B, L, N]): [description]
            labels ([B, L]): [description]
            eps ([type], optional): [description]. Defaults to 1e-10.
            padded_value_indicator (int, optional): [description]. Defaults to -100.
            reduce (str, optional): [description]. Defaults to 'mean'.
        """

        B, L, N = logits.size()
        list_labels = self.labels_to_list_label(labels, N) # B*L, N
        logits = logits.view(-1, N) #  B*L, N

        
        top1_mask = torch.eq(list_labels, 2).float()
        top2_mask = torch.eq(list_labels, 1).float()
        ignore_mask = ~torch.eq(list_labels, 2)
        ignore_mask = ignore_mask.float()
        
        max_pred_values, _ = logits.max(dim=1, keepdim=True)
        preds_minus_max = logits - max_pred_values
        
        preds_exp = preds_minus_max.exp()
        pred_exp_sum_for_top1 = torch.sum(preds_exp, dim=-1, keepdim=True)
        pred_exp_sum_for_top2 = torch.sum(preds_exp * ignore_mask, dim=-1, keepdim=True)
        
        frac_for_top1 = preds_exp / pred_exp_sum_for_top1
        frac_for_top2 = preds_exp / pred_exp_sum_for_top2
        
        log_for_top1 = torch.log(frac_for_top1 + eps) * top1_mask
        log_for_top2 = torch.log(frac_for_top2 + eps) * top2_mask
        
        loss_for_top1 = -1 * torch.sum(log_for_top1, dim=-1)
        
        top2_num = torch.sum(top2_mask, dim=-1, keepdim=True) + eps
        loss_for_top2 = log_for_top2 / top2_num
        loss_for_top2 = -1 * torch.sum(loss_for_top2, dim=-1)
        
        loss = (1. - syn_w) * loss_for_top1 + syn_w * loss_for_top2
        # loss = loss_for_top1
        
        if reduction == 'mean':
            labels_ = (labels.view(-1) != self.ignore).float()
            label_number = torch.sum(labels_, dim=-1)
            loss = loss / label_number
            loss = torch.sum(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        elif reduction is None or reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError((f'reduction: {reduction} not defined'))

        # loss_fct = nn.CrossEntropyLoss()
        # loss_ = loss_fct(logits.view(-1, N), labels.view(-1))

        # print('=====================')
        # print(labels)
        # print('---------------------')
        # print(logits)
        # print('=====================')
        
        # d = {
        #     'logits': logits.cpu(),
        #     'labels': labels.cpu(),
        # }
        # torch.save(d, 'tmp.torch')
        # print(loss.cpu().item(), loss_.cpu().item())

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, logits, labels, reduction='mean', mask=None):
        """Compute loss for model. 

        Args:
            logits: prediction scores of shape [bsz, n_class].
            labels: one-hot ground truth of shape [bsz, n_class].
        Returns:
            loss
        """
        logits = torch.div(logits, self.temperature) # B, N
        # print('logits', logits.shape)
        exp_logits = torch.exp(logits) # B, N
        # print('exp_logits', exp_logits.shape)

        if mask is not None:
            sum_exp_logits = torch.sum(exp_logits * mask, dim=-1) # B
        else:
            sum_exp_logits = torch.sum(exp_logits, dim=-1) # B
        # print('labels', labels.shape)
        # labels B x N
        exp_logits = exp_logits / sum_exp_logits.unsqueeze(1) # B x N
        log_prob = torch.log(exp_logits) * labels # B x N
        sum_log_prob = torch.sum(log_prob, dim=-1) # B

        label_number = torch.sum(labels, dim=-1) # B
        label_number = torch.clip(label_number, min=1)
        loss = -1 / label_number * sum_log_prob

        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        elif reduction is None or reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError((f'reduction: {reduction} not defined'))

        return loss
