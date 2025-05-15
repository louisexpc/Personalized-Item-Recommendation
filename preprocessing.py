import pandas as pd
import os.path as  path
import os
from tqdm import tqdm
from itertools import chain
import numpy as np
from collections import defaultdict,OrderedDict
from numpy.random import default_rng
import pickle
import random

from config import Config, load_config
def load_and_preprocessing(
        data_path:str,
        data_config: Config
    ):
    """
    Parameter:
    - data_path: path of train.csv
    - confing: object of Confing with parameters:
        - split_mode: "ratio"
            - "ratio": 按照比例切分 val 數量，需搭配 val_ratio
            - "fixed": 按照數量切分 val, 需搭配 val_n
        - val_ratio: 0.1
        - val_n : 1
        - neg_mode: "fixed"
         -neg_sample_num: 3
    """


    print("Starting data preprocessing...")
    df = pd.read_csv(data_path, dtype={
        "UserId": "str",
        "ItemId": "str"
    })

    df['UserId'] = df['UserId'].astype(int)
    df['ItemId'] = df['ItemId'].apply(lambda x: list(map(int, x.strip().split())))
    df_exploded = df.explode('ItemId').reset_index(drop=True)

    # print(df.head())
    # print(df_exploded.head())
    

    """id mapping"""
    unique_users = df_exploded['UserId'].unique()
    unique_items = df_exploded['ItemId'].unique()

    user_to_idx = {user_id: i for i, user_id in enumerate(unique_users)}
    item_to_idx = {item_id: i for i, item_id in enumerate(unique_items)}
    idx_to_user = {i: user_id for user_id, i in user_to_idx.items()}
    idx_to_item = {i: item_id for item_id, i in item_to_idx.items()}

    save_pkl(user_to_idx, os.path.join("data","user_to_idx.pkl"))
    save_pkl(item_to_idx, os.path.join("data","item_to_idx.pkl"))
    save_pkl(idx_to_user, os.path.join("data","idx_to_user.pkl"))
    save_pkl(idx_to_item, os.path.join("data","idx_to_item.pkl"))

    num_users = len(unique_users)
    num_items = len(unique_items)

    df_exploded['user_idx'] = df_exploded['UserId'].map(user_to_idx)
    df_exploded['item_idx'] = df_exploded['ItemId'].map(item_to_idx)


    print(f"Number of unique users: {num_users}")
    print(f"Number of unique items: {num_items}")
    print(f"Total interactions: {len(df_exploded)}")


    train_mapping = defaultdict(set)
    for _, row in df.iterrows():
        user_idx= user_to_idx[(row['UserId'])]
        origin_item_ids = row['ItemId']
        item_idx = [item_to_idx[item_id] for item_id in origin_item_ids]

        train_mapping[user_idx].update(set(item_idx))
    save_pkl(train_mapping, os.path.join("data","train_mapping.pkl"))
    # for uid,items in origin_preprocessd.items():
    #     print(f"{uid} (type:{type(uid)}):\n{items}")

    train_pos_dict, val_pos_dict = splitting(train_mapping=train_mapping, val_n=data_config.val_n,seed=42,split_mode=data_config.split_mode, val_ratio=data_config.val_ratio)

    bce_train_data, bpr_train_data = negative_sampling(
        train_mapping=train_mapping,
        train_pos_dict=train_pos_dict,
        num_items=num_items,
        neg_mode= data_config.neg_mode,
        neg_sample_num= data_config.neg_sample_num,
        seed=42
    )
    

    return bce_train_data,bpr_train_data,idx_to_user,idx_to_item ,num_items, num_users ,train_pos_dict,val_pos_dict,train_mapping


def negative_sampling(
        train_mapping: dict,
        train_pos_dict: dict,
        num_items :int,
        neg_mode: str = "even",
        neg_sample_num :int = 5,
        seed: int = 42
    ):
    """
    Parameters:
    - train_mapping : `defaultdict(set)` : 型態轉換完成，做完 item id mapping 並且移除重複值。 為了避免 sample 到正樣本
    - train_pos_dict: defaultdict(list) : 正樣品集
    - num_items : 總 item數量
    - neg_mode:
        - BCE:
            - "even": 每個 user 有多少個 interaction 就 sample 多少個
            - "fixed": 每個 user 指定 sample neg_sample_num 個
        - BPR:
            - "even": 一個 pos item 配一個 neg item
            - "fixed": 一個 pos item 配 neg_sample_num 個 neg item
    - neg_sample_num: 搭配 neg_mode: "fixed" 使用，指定 sample 數量
    - seed: int
    Return:
    - bce_train_data = [] # (user_idx, item_idx, label)
    - bpr_train_data = [] # (user_idx, pos_item_idx, neg_item_idx)
    """

    print("Generating negative samples for training...")
    bce_train_data = [] # (user_idx, item_idx, label)
    bpr_train_data = [] # (user_idx, pos_item_idx, neg_item_idx)

    if neg_mode == "even": print(f"Negative Sample model: {neg_mode}")
    else: print(f"Negative Sample model: {neg_mode}")

    item_pool = set(range(num_items))
    rng = default_rng(seed)

    for user_idx, items_idx_list in tqdm(train_pos_dict.items(), desc="Negative Sampling"):

        bce_train_data.extend([(user_idx,item_idx,1) for item_idx in items_idx_list])
        # For BCE
        if neg_mode == "fixed":
            bce_num_neg_sample = neg_sample_num
        else:
            bce_num_neg_sample = len(items_idx_list)

        neg_item_candidates = list(item_pool - train_mapping[user_idx])

        if len(neg_item_candidates)<= bce_num_neg_sample:
            neg_samples = neg_item_candidates
        else:
            neg_samples = rng.choice(neg_item_candidates, size=bce_num_neg_sample, replace=False)
        
        bce_train_data.extend([(user_idx, neg_item, 0) for neg_item in neg_samples])

        # For BPR
        if neg_mode == "fixed":
            bpr_num_neg_sample = neg_sample_num * len(items_idx_list)
            if len(neg_item_candidates) < bpr_num_neg_sample:
                # raise ValueError("Don't have enought to do BPR even sampling")
                print("Don't have enought to do BPR even sampling")
                continue
            

            neg_samples = rng.choice(neg_item_candidates, size=bpr_num_neg_sample, replace=False)

            for i, pos_item in enumerate(items_idx_list):
                start = i * neg_sample_num
                end   = start + neg_sample_num
                for neg in neg_samples[start:end]:
                    bpr_train_data.append((user_idx, pos_item, neg))


        else: #even
            bpr_num_neg_sample = len(items_idx_list)
            if len(neg_item_candidates) < bpr_num_neg_sample:
                raise ValueError("Don't have enought to do BPR even sampling")
            else:
                neg_samples = rng.choice(neg_item_candidates, size=bpr_num_neg_sample, replace=False)
                bpr_train_data.extend([(user_idx,items_idx_list[i],neg_samples[i]) for i in range(len(neg_samples))])

    print(f"BCE training samples: {len(bce_train_data)}")
    print(f"BPR training samples: {len(bpr_train_data)}")

    return bce_train_data, bpr_train_data
      



def splitting(train_mapping:dict, val_n : int = 5, seed:int = 42 , split_mode:str = "fixed", val_ratio:float = 0.1):
    """
    Random Pick n item for validation
    Paratmeter:
    - train_mapping : `defaultdict(set)` : 型態轉換完成，做完 item id mapping 並且移除重複值
    - val_n : 採樣數量
    - split_mode:
        - fixed:
        - ratio:

    Return:
    - train_pos_dict : defaultdict(list) without dulplicated elements
    - val_pos_dict : defaultdict(set)
    Hint: 所有切分完的 data 都是經過 mapping 的 idx
    """
    print(f"Start Spliting Train/Validation with mode :{split_mode}")
    random.seed(seed)

    train_pos_dict = defaultdict(list)
    val_pos_dict = defaultdict(set)

    for user_idx, item_idx_set in train_mapping.items():
        if split_mode == "fixed":
            if len(item_idx_set) <= val_n:
                
                train_pos_dict[user_idx] = list(item_idx_set)
            else:
                current_user_items_idx = list(item_idx_set)
                random.shuffle(current_user_items_idx)

                val_items = current_user_items_idx[-val_n:]
                train_items = current_user_items_idx[:-val_n]
                train_pos_dict[user_idx] = train_items
                
                val_pos_dict[user_idx].update(set(val_items))
        elif split_mode == "ratio":
            val_n = int(len(item_idx_set) * val_ratio)

            if val_n<=0:
                train_pos_dict[user_idx] = list(item_idx_set)
            else:
                current_user_items_idx = list(item_idx_set)
                random.shuffle(current_user_items_idx)

                val_items = current_user_items_idx[-val_n:]
                train_items = current_user_items_idx[:-val_n]
                train_pos_dict[user_idx] = train_items
                
                val_pos_dict[user_idx].update(set(val_items))
        else:
            raise ValueError(f"Spliting mode {split_mode} not found, shoud be 'fixed' or 'ratio' .")
    
    return train_pos_dict, val_pos_dict


        
def save_pkl(file,path):
    try:
        with open(path, "wb") as f:
            pickle.dump(file, f) 
    except Exception as e:
        print(f"Save file error: {e}")

def load_pickle(self,path):
        try:
            with open(path,'rb') as f:
                target = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Loading pkl failed: {e}")
        return target

if __name__ == '__main__':
    data_config = load_config(r"config\data.yaml")
    load_and_preprocessing(r"data\train.csv",data_config)