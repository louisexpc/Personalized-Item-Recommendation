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

from model import BCEDataset,BPRDataset,BPRLoss,BCELoss

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