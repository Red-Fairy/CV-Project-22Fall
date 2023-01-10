import torch
from torch import nn

def pair_cosine_similarity(x, y=None, eps=1e-8):
    if(y == None):
        n = x.norm(p=2, dim=1, keepdim=True)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps)
    else:
        n1 = x.norm(p=2, dim=1, keepdim=True)
        n2 = y.norm(p=2, dim=1, keepdim=True)
        return (x @ y.t()) / (n1 * n2.t()).clamp(min=eps)


def nt_xent(x, y=None, t=0.5):
    if y is not None:
        x = pair_cosine_similarity(x, y)
    else:
        x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x).mean()

# def sup_nt_xent(x, label, t=0.5):

#     anchor_dot_contrast = torch.div(torch.matmul(x, x.T), t)

#     print(anchor_dot_contrast.mean())
#     logits = anchor_dot_contrast

#     print(logits.mean())

#     batch_size = x.shape[0]

#     label1 = label.view(-1, 1).expand_as(anchor_dot_contrast)
#     label2 = label.view(1, -1).expand_as(anchor_dot_contrast)
#     # tile mask
#     mask = (label1 == label2).float()

#     # mask-out self-contrast cases
#     logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size).view(-1, 1).cuda(),0)
#     mask = mask * logits_mask

#     # compute log_prob
#     exp_logits = torch.exp(logits) * logits_mask

#     print(exp_logits.mean())
#     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#     print(log_prob.mean())

#     # compute mean of log-likelihood over positive
#     mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#     print(mean_log_prob_pos.mean())

#     # loss
#     loss = - mean_log_prob_pos
#     loss = loss.view(1, batch_size).mean()

#     exit()

#     print(loss)

#     return loss

# def sup_nt_xent(x, label, t=0.5):

#     label = label.detach()
#     x = pair_cosine_similarity(x)
#     x = torch.exp(x / t)
#     label1 = label.view(-1, 1).expand_as(x)
#     label2 = label.view(1, -1).expand_as(x)

#     mask = (label1 == label2).float() - torch.eye(x.size()[0]).cuda()
#     # print(mask)
#     positive = mask.sum(0)
#     # print(positive)

#     y = ((x * mask) / (x.sum(0) - torch.exp(torch.tensor(1 / t))).expand(x.size()[0], x.size()[0])) + 1e-6

#     # exit()

#     return ((-torch.log(y) * mask).sum(0) / positive).mean()

def sup_nt_xent(x, label, t=0.5):
    label = label.detach()
    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    label1 = label.view(-1, 1).expand_as(x)
    label2 = label.view(1, -1).expand_as(x)
    mask = (label1 != label2).float() + torch.eye(x.size()[0]).cuda()

    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]

    x = x * mask

    x = x.diag() / x.sum(0)
    return -torch.log(x).mean()

def nt_xent3(x, t=0.5):
    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    idx[::3] += 1
    idx[1::3] += 1
    idx[2::3] -= 2
    x = x[idx] # rotation
    loss = 0.
    loss += -torch.log(x.diag()/(x.sum(0) - torch.exp(torch.tensor(1 / t)))).mean()
    x = x[idx]
    loss += -torch.log(x.diag()/(x.sum(0) - torch.exp(torch.tensor(1 / t)))).mean()

    return loss / 2 # divide by 2 to get the average loss

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = weight.cuda()
        self.gamma = gamma

    def forward(self, input, target):
        """
        input: [N, C], logits before softmax
        target: [N], labels
        """
        y_pred = torch.softmax(input, dim=1)
        ce = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='none')(input, target)
        # get the target probability
        p = y_pred.gather(1, target.view(-1, 1)).squeeze()
        loss = ce * (1 - p) ** self.gamma
        return loss.mean()

class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets, ):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)
        batch_cls_count = torch.eye(len(self.cls_num_list))[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)
        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + len(self.cls_num_list)) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss