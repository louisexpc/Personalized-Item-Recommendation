import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm # 用於顯示進度條
import random
import time
from typing import DefaultDict,Set, List
from numpy.random import default_rng

from model import BCEDataset,BPRDataset,BPRLoss,BCELoss, PairWiseDataset

def train_bpr_model_with_hard_negative_sampling_vectorize(
        model,
        train_mapping :DefaultDict[int, Set],
        train_pos_dict: DefaultDict[int, List], 
        val_pos_dict: DefaultDict[int, Set] ,
        num_items:int,
        train_config,
    ):
    # Initialization
    num_epochs = train_config.epochs
    lr = train_config.lr
    batch_size = train_config.batch_size
    weight_decay = train_config.weight_decay
    top_k = train_config.top_k
    device = torch.device(train_config.device)
    """Update: Hard Negative Sampling"""
    num_candidate_neg_samples = train_config.num_candidate_neg_samples
    num_hard_neg_samples = train_config.num_hard_neg_samples
    num_random_neg_samples = train_config.num_random_neg_samples
    rng = default_rng(42)
    

    """Prepare Dataset """
    train_pos_list = train_pos_list = [
        (user_idx, item_idx)
        for user_idx, items_idx_list_for_user in train_pos_dict.items()
        for item_idx in items_idx_list_for_user
    ]

    print(f"BPR hard negative sampling :\n Number of Pos Interactions: {len(train_pos_list)}")
    
    dataset = PairWiseDataset(train_pos_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion =  BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Added weight_decay for regularization

    all_item_idx_list = set(range(num_items))

    model.to(device)
    best_map = 0.0
    best_state = None

    patience = 5
    counter = 0
    print("\nTraining BPR Model with hard negative sampling vectorization")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        for users, pos_items in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_size = users.size(0)
            optimizer.zero_grad()

            # neg_items_index = torch.randint(0, num_items, (batch_size, num_candidate_neg_samples), device=device) #(B,M)
            neg_items_index = []
            for user_idx in users.tolist():
                neg_candidate = list(all_item_idx_list - set(train_mapping[int(user_idx)]))
                neg_items = rng.choice(neg_candidate,size=num_candidate_neg_samples,replace=False)
                neg_items_index.append(neg_items)
            
            neg_items_index = np.array(neg_items_index)
            neg_items_index = torch.tensor(neg_items_index,dtype=torch.long, device= device)

            users_expanded = users.unsqueeze(1).expand(-1, num_candidate_neg_samples) #(B,M)

            
            with torch.no_grad():
                users_expanded_flatten = users_expanded.reshape(-1) #(B*M,)
                neg_items_index_flatten = neg_items_index.reshape(-1)#(B*M,)
                candidate_neg_scores = model(users_expanded_flatten,neg_items_index_flatten).view(batch_size, num_candidate_neg_samples) #(B,M)

            actual_k_hard = min(num_hard_neg_samples, num_candidate_neg_samples)
            _, hard_indices_in_candidates = torch.topk(candidate_neg_scores, k=actual_k_hard, dim=1) # (B, N_hard)
            hard_neg_items = torch.gather(neg_items_index, 1, hard_indices_in_candidates) # (B, N_hard)

            if num_random_neg_samples > 0:
                #random_selection_indices = torch.randint(0, num_candidate_neg_samples, (batch_size, num_random_neg_samples), device=device)
                random_selection_index =  []
                for i,user_idx in enumerate(users.tolist()):
                    random_neg_candidate = list(all_item_idx_list - train_mapping[int(user_idx)] - set(hard_neg_items[i,:].tolist()))
                    random_neg_items = rng.choice(random_neg_candidate,size= num_random_neg_samples,replace=False)
                    random_selection_index.append(random_neg_items)
                random_selection_index = np.array(random_selection_index)
                random_selection_index = torch.tensor(random_selection_index,dtype=torch.long, device= device)

                #random_neg_items = torch.gather(neg_items_index, 1, random_selection_index) # (B, N_random)
                final_selected_neg_items = torch.cat([hard_neg_items, random_selection_index], dim=1) # (B, N_hard + N_random)
            else:
                final_selected_neg_items = hard_neg_items # (B, N_hard)

            
            num_selected_negs_per_pos = final_selected_neg_items.size(1)
            model.train()
            optimizer.zero_grad()
            pos_scores = model(users, pos_items) # (B)
            pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, num_selected_negs_per_pos) # (B, N_hard + N_random)

            users_final_expanded = users.unsqueeze(1).expand(-1, num_selected_negs_per_pos) 

            selected_neg_scores = model(
                users_final_expanded.reshape(-1),
                final_selected_neg_items.reshape(-1)
            ).view(batch_size, num_selected_negs_per_pos)

            loss = criterion(pos_scores_expanded.reshape(-1), selected_neg_scores.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - start_time

        if (epoch + 1) % 1 == 0 :
            val_map_score = map_at_k(
                model=model,
                train_pos_dict=train_pos_dict,
                val_pos_dict=val_pos_dict,
                num_items=num_items,
                device=device,
                top_k=top_k
            )
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Val MAP@{top_k}: {val_map_score:.4f}, Time: {epoch_duration:.2f}s")

            if val_map_score > best_map:
                best_map, best_state = val_map_score, model.state_dict()
                counter = 0
            else:
                counter+=1

            if  counter >= patience:
                print(f"Early Stopping at epoch {epoch}, Best Val MAP@{top_k}: {best_map:.4f}")
                break
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Time: {epoch_duration:.2f}s")
        
    model.load_state_dict(best_state)

    return model



def train_bpr_model_with_hard_negative_sampling(
        model,
        train_mapping :DefaultDict[int, Set],
        train_pos_dict: DefaultDict[int, List], 
        val_pos_dict: DefaultDict[int, Set] ,
        num_items:int,
        train_config,
    ):
    # Initialization
    num_epochs = train_config.epochs
    lr = train_config.lr
    batch_size = train_config.batch_size
    weight_decay = train_config.weight_decay
    top_k = train_config.top_k
    device = torch.device(train_config.device)
    """Update: Hard Negative Sampling"""
    num_candidate_neg_samples = train_config.num_candidate_neg_samples
    num_hard_neg_samples = train_config.num_hard_neg_samples
    num_random_neg_samples = train_config.num_random_neg_samples
    

    """Prepare Dataset """
    train_pos_list = train_pos_list = [
        (user_idx, item_idx)
        for user_idx, items_idx_list_for_user in train_pos_dict.items()
        for item_idx in items_idx_list_for_user
    ]

    print(f"BPR hard negative sampling :\n Number of Pos Interactions: {len(train_pos_list)}")
    
    dataset = PairWiseDataset(train_pos_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion =  BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Added weight_decay for regularization

    all_item_idx_list = list(range(num_items))

    model.to(device)
    best_map = 0.0
    best_state = None

    patience = 5
    counter = 0
    print("\nTraining BPR Model with hard negative sampling")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        for users, pos_items in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            users, pos_items = users.to(device), pos_items.to(device)
            """Hard negative sampling for the positive pair (user, pos_item)"""
           
            batch_size = users.size(0)
            
            selected_neg_items_list = []

            optimizer.zero_grad()

            model.eval()

            with torch.no_grad():
                for i in range(batch_size):
                    user_idx = users[i].item()
                    
                    user_known_positives = set(train_mapping[user_idx])

                    current_candidate_neg_items = []
                    attempts = 0
                    while len(current_candidate_neg_items) < num_candidate_neg_samples  and attempts < num_candidate_neg_samples* 5:
                        sampled_item_idx = random.choice(all_item_idx_list)
                        if sampled_item_idx not in user_known_positives: # 確保不是該用戶任何已知的正樣本
                            current_candidate_neg_items.append(sampled_item_idx)
                        attempts += 1

                    candidate_neg_items_tensor = torch.tensor(current_candidate_neg_items, dtype=torch.long).to(device)
                    user_tensor_expanded = users[i].unsqueeze(0).expand(len(current_candidate_neg_items)) # user_idx 重複 M 次

                    candidate_neg_scores = model(user_tensor_expanded, candidate_neg_items_tensor)

                    actual_k_hard = min(num_hard_neg_samples, len(current_candidate_neg_items))
                    _, hard_indices = torch.topk(candidate_neg_scores, k=actual_k_hard)
                    hard_neg_items = candidate_neg_items_tensor[hard_indices]

                    remaining_indices = [idx for idx in range(len(current_candidate_neg_items)) if idx not in hard_indices.tolist()]
                    actual_k_random = min(num_random_neg_samples, len(remaining_indices))

                    if actual_k_random > 0:
                        random_selection_indices = random.sample(remaining_indices, k=actual_k_random)
                        random_neg_items = candidate_neg_items_tensor[random_selection_indices]
                        final_neg_for_user = torch.cat([hard_neg_items, random_neg_items])
                    else: # 如果沒有剩餘的可以選作random，或NUM_RANDOM_NEG_SAMPLES=0
                        final_neg_for_user = hard_neg_items

                    if len(final_neg_for_user) == 0 and len(current_candidate_neg_items) > 0: # 如果hard和random都沒選到，但有候選
                        final_neg_for_user = candidate_neg_items_tensor[0:1] # 保底選一個
                    
                    selected_neg_items_list.append(final_neg_for_user)
            
            
            model.train()
            optimizer.zero_grad()

            pos_scores_bpr = model(users, pos_items) # (batch_size)

            expanded_pos_scores = []
            all_selected_neg_scores = []
            for i in range(batch_size):
                num_selected_negs = len(selected_neg_items_list[i])
                if num_selected_negs == 0: continue

                # 重複正樣本得分
                expanded_pos_scores.append(pos_scores_bpr[i].expand(num_selected_negs))
                
                # 計算對應的負樣本得分
                user_tensor_expanded_loss = users[i].unsqueeze(0).expand(num_selected_negs)
                current_neg_scores = model(user_tensor_expanded_loss, selected_neg_items_list[i])
                all_selected_neg_scores.append(current_neg_scores)

            if not expanded_pos_scores: # 如果沒有有效的負樣本被選出
                loss_bpr = torch.tensor(0.0, device=device, requires_grad=True) # 或者跳過這個batch
            else:
                final_pos_scores = torch.cat(expanded_pos_scores) # (total_selected_negs)
                final_neg_scores = torch.cat(all_selected_neg_scores) # (total_selected_negs)
                loss_bpr = criterion(final_pos_scores, final_neg_scores)
            
            loss_bpr.backward()
            optimizer.step()
            epoch_loss += loss_bpr.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - start_time

        if (epoch + 1) % 1 == 0 :
            val_map_score = map_at_k(
                model=model,
                train_pos_dict=train_pos_dict,
                val_pos_dict=val_pos_dict,
                num_items=num_items,
                device=device,
                top_k=top_k
            )
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Val MAP@{top_k}: {val_map_score:.4f}, Time: {epoch_duration:.2f}s")

            if val_map_score > best_map:
                best_map, best_state = val_map_score, model.state_dict()
                counter = 0
            else:
                counter+=1

            if  counter >= patience:
                print(f"Early Stopping at epoch {epoch}, Best Val MAP@{top_k}: {best_map:.4f}")
                break
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Time: {epoch_duration:.2f}s")
        
    model.load_state_dict(best_state)

    return model




def train_bce_model(model, train_data,train_pos_dict,val_pos_dict,num_items, train_config):
    # Initialization
    num_epochs = train_config.epochs
    lr = train_config.lr
    batch_size = train_config.batch_size
    weight_decay = train_config.weight_decay
    top_k = train_config.top_k
    device = torch.device(train_config.device)

    dataset = BCEDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion =  BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Added weight_decay for regularization

    model.to(device)

    """Update : Best Model Saving and Early Stop"""
    best_map = 0.0
    best_state = None

    patience = 5
    counter = 0

    print("\nTraining BCE Model...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for users, items, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            users, items, labels = users.to(device), items.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - start_time

        if (epoch + 1) % 1 == 0 :
            val_map_score = map_at_k(
                model=model,
                train_pos_dict=train_pos_dict,
                val_pos_dict=val_pos_dict,
                num_items=num_items,
                device=device,
                top_k=top_k
            )
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Val MAP@{top_k}: {val_map_score:.4f}, Time: {epoch_duration:.2f}s")

            if val_map_score > best_map:
                best_map, best_state = val_map_score, model.state_dict()
                counter = 0
            else:
                counter+=1

            if  counter >= patience:
                print(f"Early Stopping at epoch {epoch}, Best Val MAP@{top_k}: {best_map:.4f}")
                break
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Time: {epoch_duration:.2f}s")
        
    model.load_state_dict(best_state)

    return model

def train_bpr_model(model, train_data,train_pos_dict,val_pos_dict,num_items, train_config):
    # Initialization
    num_epochs = train_config.epochs
    lr = train_config.lr
    batch_size = train_config.batch_size
    weight_decay = train_config.weight_decay
    top_k = train_config.top_k
    margin = train_config.margin
    
    device = torch.device(train_config.device)

    dataset = BPRDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = BPRLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    """Update : Best Model Saving and Early Stop"""
    best_map = 0.0
    best_state = None

    patience = 5
    counter = 0

    print("\nTraining BPR Model...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for users, pos_items, neg_items in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            pos_scores = model(users, pos_items)
            neg_scores = model(users, neg_items)

            loss = criterion(pos_scores,neg_scores)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - start_time
        if (epoch + 1) % 1 == 0: # Evaluate every epoch
            val_map_score = map_at_k(
                model=model,
                train_pos_dict=train_pos_dict,
                val_pos_dict=val_pos_dict,
                num_items=num_items,
                device=device,
                top_k=top_k
            )
            
            print(f"Epoch {epoch+1}/{num_epochs}, BPR Loss: {avg_epoch_loss:.4f}, Val MAP@{top_k}: {val_map_score:.4f}, Time: {epoch_duration:.2f}s")

            if val_map_score > best_map:
                best_map, best_state = val_map_score, model.state_dict()
                counter = 0
            else:
                counter+=1

            if  counter >= patience:
                print(f"Early Stopping at epoch {epoch}, Best Val MAP@{top_k}: {best_map:.4f}")
                break
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, BPR Loss: {avg_epoch_loss:.4f}, Time: {epoch_duration:.2f}s")

    model.load_state_dict(best_state)
    return model

def average_precision_at_k(actual_items, predicted_item_scores, k):
    """
    Computes Average Precision at K.
    Parameters
    - actual_items: set(int), user 真正接過過的 item
    - predicted_item_scores: A list of (item_idx, score) tuples, sorted by score descending.
    - k: int. The number of items to consider in the predicted list.
    """
    if not actual_items:
        return 0.0
    predicted_items_at_k = [item_idx for item_idx, score in predicted_item_scores[:k]]
    
    score = 0.0
    num_hits = 0.0
    for i, p_item_idx in enumerate(predicted_items_at_k):
        if p_item_idx in actual_items:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    if num_hits == 0: 
        return 0.0
        
    return score / min(len(actual_items), k)


def map_at_k(model, train_pos_dict, val_pos_dict, num_items,device, top_k):
    """
    Measure all validation dataset at training
    Parameter:
    - model : LFM
    - train_pos_dict : defaultdict(list) with out dulplicated elements
    - val_pos_dict : defaultdict(set)
    - num_items: for generate item idx set
    - device: cpu or gpu (only cpu here)
    - top_k: int, compute MAP@K
    """

    model.eval() # Set model to evaluation mode
    aps = [] # List to store Average Precisions for each user

    all_item_idx_set = set(range(num_items))
    
    with torch.no_grad():
        for user_idx, true_val_items_set in tqdm(val_pos_dict.items(), desc="Calculating MAP@k", leave=False):

            if not true_val_items_set: 
                continue
            user_tensor = torch.tensor([int(user_idx)], dtype=torch.long).to(device)
            items_to_exclude_for_user = set(train_pos_dict[user_idx])

            candidate_item_idx = list(all_item_idx_set - items_to_exclude_for_user)

            #edge case : empty
            if not candidate_item_idx: 
                continue

            candidate_item_idx_tensor = torch.tensor(candidate_item_idx, dtype=torch.long).to(device)
            scores = model.predict_score(user_tensor, candidate_item_idx_tensor)

            if scores.ndim == 0: # if only one candidate item, scores might be scalar
                scores = scores.unsqueeze(0)
            # item_scores : [(item_idx, score)]
            item_scores = []
            for i, item_internal_idx in enumerate(candidate_item_idx):
                item_scores.append((item_internal_idx, scores[i].item()))
            
            item_scores.sort(key=lambda x: x[1], reverse=True)

            ap = average_precision_at_k(true_val_items_set, item_scores, top_k)
            aps.append(ap)

    if not aps:
        return 0.0
    return np.mean(aps)

def predict_top_k_for_all_users(
        model,train_mapping, num_items, num_users,idx_to_user,idx_to_item,device, top_k
    ):
    """

    Parameters:
    - train_mapping: defaultdict(set), user training item idx
    """

    model.eval()
    predictions = []
    all_item_idx_set = set(range(num_items))

    print("\nGenerating predictions for all users...")
    for user_idx in tqdm(range(num_users), desc="Predicting"):
        original_user_id = idx_to_user[user_idx]

        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)
        
        exist_user_item_idx_set = train_mapping[user_idx] 
        candidate_item_idx = list(all_item_idx_set - exist_user_item_idx_set)

        if not candidate_item_idx:
            predictions.append({'UserId': original_user_id, 'ItemId': ""})
            continue

        candidate_item_idx_tensor = torch.tensor(candidate_item_idx,dtype=torch.long).to(device)

        with torch.no_grad():
            scores = model.predict_score(user_tensor, candidate_item_idx_tensor)
            if scores.ndim == 0: # if only one candidate item
                scores = scores.unsqueeze(0)
        
        item_scores = []
        for i, item_internal_idx in enumerate(candidate_item_idx):
            item_scores.append((item_internal_idx, scores[i].item()))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_item_internal_indices = [item_idx for item_idx, score in item_scores[:top_k]]
        top_k_item_original_ids = [idx_to_item[idx] for idx in top_k_item_internal_indices]
        top_k_item_original_ids_str = " ".join(map(str, top_k_item_original_ids))
        
        predictions.append({'UserId': original_user_id, 'ItemId': top_k_item_original_ids_str})
    return pd.DataFrame(predictions)