import torch
from torch import nn


class Bpr(nn.Module):
    def __init__(self, user_num, item_num, dim):
        super(Bpr, self).__init__()
        self.user_embedding = nn.Embedding(user_num, dim)
        self.item_embedding = nn.Embedding(item_num, dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users, pos_items, neg_items):
        users_emb = self.user_embedding(users)
        pos_items_emb = self.item_embedding(pos_items)
        neg_items_emb = self.item_embedding(neg_items)
        u_i = torch.sum(users_emb * pos_items_emb, 1)
        u_j = torch.sum(users_emb * neg_items_emb, 1)
        loss = -(torch.sum(torch.log2(torch.sigmoid(u_i - u_j))))
        return loss

    def predict(self, user, pos_item, neg_item):
        user_emb = self.user_embedding(user)
        pos_item_emb = self.item_embedding(pos_item)
        neg_item_emb = self.item_embedding(neg_item)
        u_i = torch.sum(user_emb * pos_item_emb, 1)
        u_j = torch.sum(user_emb * neg_item_emb, 1)
        return u_i-u_j

