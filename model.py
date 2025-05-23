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

class PairWiseDataset(Dataset):
    def __init__(self, data):
    
        self.users = torch.tensor([d[0] for d in data], dtype=torch.long)
        self.pos_items = torch.tensor([d[1] for d in data], dtype=torch.long)
        

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx]

# class LFM(nn.Module):
#     def __init__(self, num_users:int, num_items:int, num_factors:int):
#         super(LFM, self).__init__()
#         self.user_factors = nn.Embedding(num_users, num_factors)
#         self.item_factors = nn.Embedding(num_items, num_factors)

        
#         nn.init.normal_(self.user_factors.weight, std=0.01)
#         nn.init.normal_(self.item_factors.weight, std=0.01)

#     def forward(self, user_indices, item_indices):
#         user_vecs = self.user_factors(user_indices)
#         item_vecs = self.item_factors(item_indices)
        
#         return (user_vecs * item_vecs).sum(dim=1)

#     def predict_score(self, user_idx_tensor, item_idx_tensor):
#         user_vec = self.user_factors(user_idx_tensor) 
#         item_vecs = self.item_factors(item_idx_tensor) 
        
#         if user_vec.ndim == 2 and item_vecs.ndim == 2:
#             if user_vec.shape[0] == 1 and item_vecs.shape[0] > 1: 
#                  return torch.matmul(user_vec, item_vecs.transpose(0, 1)).squeeze()
#             elif item_vecs.shape[0] == 1 and user_vec.shape[0] > 1: 
#                  return torch.matmul(item_vecs, user_vec.transpose(0,1)).squeeze() 
#             elif user_vec.shape[0] == item_vecs.shape[0]:
#                 return (user_vec * item_vecs).sum(dim=1)

#         if user_vec.ndim == 1: 
#             user_vec = user_vec.unsqueeze(0) 
        
#         return torch.matmul(user_vec, item_vecs.transpose(0, 1)).squeeze()


class LFM(nn.Module):
    def __init__(self, num_users:int, num_items:int, num_factors:int):
        super(LFM, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)

        """Update bias"""
        self.user_biases = nn.Embedding(num_users, 1) 
        self.item_biases = nn.Embedding(num_items, 1) 
        self.global_bias = nn.Parameter(torch.zeros(1)) 


        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

        """Update: Init bias"""
        nn.init.zeros_(self.user_biases.weight) 
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, user_indices, item_indices):
        user_vecs = self.user_factors(user_indices)
        item_vecs = self.item_factors(item_indices)
        
        user_b = self.user_biases(user_indices).squeeze() 
        item_b = self.item_biases(item_indices).squeeze() 
        
        dot_product = (user_vecs * item_vecs).sum(dim=1) 
        
        return self.global_bias + user_b + item_b + dot_product

    def predict_score(self, user_idx_tensor, item_idx_tensor):
      
        user_vecs = self.user_factors(user_idx_tensor)
        item_vecs = self.item_factors(item_idx_tensor)
        
        user_b = self.user_biases(user_idx_tensor) 
        item_b = self.item_biases(item_idx_tensor)


        if user_vecs.ndim == 1: user_vecs = user_vecs.unsqueeze(0)
        if item_vecs.ndim == 1: item_vecs = item_vecs.unsqueeze(0)
        if user_b.ndim == 1: user_b = user_b.unsqueeze(1) 
        if item_b.ndim == 1: item_b = item_b.unsqueeze(1) 


        if user_vecs.shape[0] == 1 and item_vecs.shape[0] > 1: 
            dot_product = torch.matmul(user_vecs, item_vecs.transpose(0, 1)).squeeze() 
            
            scores = self.global_bias + user_b.squeeze() + item_b.squeeze() + dot_product
            return scores
  
        elif user_vecs.shape[0] == item_vecs.shape[0]:
             dot_product = (user_vecs * item_vecs).sum(dim=1) # (N)
             scores = self.global_bias + user_b.squeeze() + item_b.squeeze() + dot_product
             return scores
        else: 
            if user_idx_tensor.ndim == 0: user_idx_tensor = user_idx_tensor.unsqueeze(0)
            if item_idx_tensor.ndim == 0: item_idx_tensor = item_idx_tensor.unsqueeze(0)
            
            user_vecs = self.user_factors(user_idx_tensor) 
            item_vecs = self.item_factors(item_idx_tensor) 
            user_b_val = self.user_biases(user_idx_tensor).squeeze() 
            item_b_vals = self.item_biases(item_idx_tensor).squeeze() 
            
            dot_product = torch.matmul(user_vecs, item_vecs.transpose(0,1)).squeeze() 
            return self.global_bias + user_b_val + item_b_vals + dot_product


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
    """
    Parameters:
    - margin: float, 拉高 pos item score
    """
    def __init__(self, margin = 1.0): 
        super(BPRLoss, self).__init__() 
        self.margin = margin

    def forward(self, pos_score, neg_score):
        """
        Parameters:
        - pos_score: tensor [batch_size]
        - neg_score: tensor [batch_size]
        """
        diff = pos_score - neg_score - self.margin
        log_likelihood = F.logsigmoid(diff) 


        final_loss = -log_likelihood.mean()

        return final_loss

