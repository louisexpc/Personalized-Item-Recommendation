from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDataset(Dataset):
    def __init__(self, data):

        self.users = torch.tensor([d[0] for d in data], dtype=torch.long)
        self.items = torch.tensor([d[1] for d in data], dtype=torch.long)
        self.labels = torch.tensor([d[2] for d in data], dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

class BPRDataset(Dataset):
    def __init__(self, data):
    
        self.users = torch.tensor([d[0] for d in data], dtype=torch.long)
        self.pos_items = torch.tensor([d[1] for d in data], dtype=torch.long)
        self.neg_items = torch.tensor([d[2] for d in data], dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]

class LFM(nn.Module):
    def __init__(self, num_users:int, num_items:int, num_factors:int):
        super(LFM, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)

        
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        user_vecs = self.user_factors(user_indices)
        item_vecs = self.item_factors(item_indices)
        
        return (user_vecs * item_vecs).sum(dim=1)

    def predict_score(self, user_idx_tensor, item_idx_tensor):
        user_vec = self.user_factors(user_idx_tensor) 
        item_vecs = self.item_factors(item_idx_tensor) 
        
        if user_vec.ndim == 2 and item_vecs.ndim == 2:
            if user_vec.shape[0] == 1 and item_vecs.shape[0] > 1: 
                 return torch.matmul(user_vec, item_vecs.transpose(0, 1)).squeeze()
            elif item_vecs.shape[0] == 1 and user_vec.shape[0] > 1: 
                 return torch.matmul(item_vecs, user_vec.transpose(0,1)).squeeze() 
            elif user_vec.shape[0] == item_vecs.shape[0]:
                return (user_vec * item_vecs).sum(dim=1)

        if user_vec.ndim == 1: 
            user_vec = user_vec.unsqueeze(0) 
        
        return torch.matmul(user_vec, item_vecs.transpose(0, 1)).squeeze()


class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()

    def forward(self, prediction, Label):
        
        # print(f'input:{type(prediction)}:{prediction}')
        # print(f"label:{type(Label)}:{Label}")
        log_inputs = torch.clamp(torch.log(torch.sigmoid(prediction)), min=-100.0)
        log_one_minus_inputs = torch.clamp(torch.log(1 - torch.sigmoid(prediction)), min=-100.0)

        
        single_loss = - (Label * log_inputs + (1 - Label) * log_one_minus_inputs)
        final_loss = single_loss.mean()
        return final_loss


class BPRLoss(nn.Module): 
    def __init__(self, weight=None): 
        super(BPRLoss, self).__init__() 

    def forward(self, pos_score, neg_score):
        """
        Parameters:
        - pos_score: tensor [batch_size]
        - neg_score: tensor [batch_size]
        """
        diff = pos_score - neg_score
        log_likelihood = F.logsigmoid(diff) 


        final_loss = -log_likelihood.mean()

        return final_loss

