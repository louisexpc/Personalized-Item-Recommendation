# Personalized-Item-Recommendation

## Homework Link
- [Kaggle](https://www.kaggle.com/competitions/wm-2025-personalized-item-recommendation/data)
- [Slides](https://docs.google.com/presentation/d/1Zhs0z3hr26DoYUl0TovZ7T5A5uBa2I-qP8wd5KOFv3s/edit?slide=id.p#slide=id.p)
- [Announcements](https://cool.ntu.edu.tw/courses/44637/discussion_topics/395702)

## Data Preprocessing
- `train_mapping.pkl`: `defaultdict(set)` : 型態轉換完成，做完 item id mapping 並且移除重複值
- view:
    ```python
    # --- 在您的訓練函數 (例如 train_bpr_model) 內部 ---

    # 新增超參數
    NUM_CANDIDATE_NEG_SAMPLES = 10 # M: 每個正樣本對應的候選負樣本數
    NUM_HARD_NEG_SAMPLES = 2     # N_hard
    NUM_RANDOM_NEG_SAMPLES = 2   # N_random
    # 確保 NUM_HARD_NEG_SAMPLES + NUM_RANDOM_NEG_SAMPLES > 0

    # ... 進入 epoch 迴圈 ...
    # ... 進入 batch 迴圈 ...
    # users, pos_items = batch # 從 DataLoader 獲取
    # users, pos_items = users.to(DEVICE), pos_items.to(DEVICE)

    # optimizer.zero_grad()

    # 1. 計算正樣本得分
    # pos_scores = model(users, pos_items) # (batch_size)

    # 2. 為批次中的每個 (user, pos_item) 進行 Hard Negative Sampling
    #    all_item_indices_list: 所有 item 內部索引的列表
    #    train_user_positive_items_idx: dict, {user_idx: {pos_item_idx1, ...}}

    batch_size = users.size(0)
    selected_neg_items_list = [] # 儲存為每個正樣本選出的負樣本

    # model.eval() # 可選：如果不想BN等層在負採樣時更新狀態，但通常MF模型沒有這些
    with torch.no_grad(): # 不計算負樣本評分的梯度
        for i in range(batch_size):
            user_idx = users[i].item()
            current_pos_item_idx = pos_items[i].item() # 當前正樣本，理論上負樣本不應包含它
            
            user_known_positives = train_user_positive_items_idx.get(user_idx, set())
            
            current_candidate_neg_items = []
            attempts = 0
            while len(current_candidate_neg_items) < NUM_CANDIDATE_NEG_SAMPLES and attempts < NUM_CANDIDATE_NEG_SAMPLES * 5:
                sampled_item_idx = random.choice(all_item_indices_list)
                if sampled_item_idx not in user_known_positives: # 確保不是該用戶任何已知的正樣本
                    current_candidate_neg_items.append(sampled_item_idx)
                attempts += 1
            # 如果採樣不足，用隨機樣本補足（可能包含已知正例之外的任意樣本，需小心處理）
            # 實務上，如果物品夠多，通常能採到足夠的
            if not current_candidate_neg_items: # 極端情況，無法採樣到任何負例
                # 簡單處理：隨機選一些不等於 current_pos_item_idx 的 (可能不是最優)
                current_candidate_neg_items = random.sample(
                    [item for item in all_item_indices_list if item != current_pos_item_idx],
                    k=NUM_CANDIDATE_NEG_SAMPLES
                )


            candidate_neg_items_tensor = torch.tensor(current_candidate_neg_items, dtype=torch.long).to(DEVICE)
            user_tensor_expanded = users[i].unsqueeze(0).expand(len(current_candidate_neg_items)) # user_idx 重複 M 次
            
            # 獲取候選負樣本的評分
            candidate_neg_scores = model(user_tensor_expanded, candidate_neg_items_tensor) # (M)
            
            # 選擇 Hard Negatives
            # _, hard_indices = torch.topk(candidate_neg_scores, k=min(NUM_HARD_NEG_SAMPLES, len(current_candidate_neg_items)))
            # 確保 k 不大於候選樣本數
            actual_k_hard = min(NUM_HARD_NEG_SAMPLES, len(current_candidate_neg_items))
            _, hard_indices = torch.topk(candidate_neg_scores, k=actual_k_hard)

            hard_neg_items = candidate_neg_items_tensor[hard_indices]
            
            # 選擇 Random Negatives (從剩餘的候選中)
            remaining_indices = [idx for idx in range(len(current_candidate_neg_items)) if idx not in hard_indices.tolist()]
            actual_k_random = min(NUM_RANDOM_NEG_SAMPLES, len(remaining_indices))
            if actual_k_random > 0:
                random_selection_indices = random.sample(remaining_indices, k=actual_k_random)
                random_neg_items = candidate_neg_items_tensor[random_selection_indices]
                final_neg_for_user = torch.cat([hard_neg_items, random_neg_items])
            else: # 如果沒有剩餘的可以選作random，或NUM_RANDOM_NEG_SAMPLES=0
                final_neg_for_user = hard_neg_items
            
            if len(final_neg_for_user) == 0 and len(current_candidate_neg_items) > 0: # 如果hard和random都沒選到，但有候選
                final_neg_for_user = candidate_neg_items_tensor[0:1] # 保底選一個
            elif len(final_neg_for_user) == 0 and len(current_candidate_neg_items) == 0:
                # 極端情況：真的沒有任何負例可以選，這不應該發生
                # 這裡需要一個 fallback，例如隨機選一個不是 positive 的 item
                fallback_neg_item = random.choice([item for item in all_item_indices_list if item != current_pos_item_idx])
                final_neg_for_user = torch.tensor([fallback_neg_item], dtype=torch.long).to(DEVICE)


            selected_neg_items_list.append(final_neg_for_user)

    # model.train() # 恢復訓練模式

    # 現在 selected_neg_items_list 包含了每個正樣本對應的一組精選負樣本
    # 需要將其轉換為適合損失函數計算的格式

    # === 對於 BPR Loss ===
    # optimizer.zero_grad() # 移到這裡或更前面
    # pos_scores_bpr = model(users, pos_items) # (batch_size)

    # expanded_pos_scores = []
    # all_selected_neg_scores = []
    # for i in range(batch_size):
    #     num_selected_negs = len(selected_neg_items_list[i])
    #     if num_selected_negs == 0: continue

    #     # 重複正樣本得分
    #     expanded_pos_scores.append(pos_scores_bpr[i].expand(num_selected_negs))
        
    #     # 計算對應的負樣本得分
    #     user_tensor_expanded_loss = users[i].unsqueeze(0).expand(num_selected_negs)
    #     current_neg_scores = model(user_tensor_expanded_loss, selected_neg_items_list[i])
    #     all_selected_neg_scores.append(current_neg_scores)

    # if not expanded_pos_scores: # 如果沒有有效的負樣本被選出
    #     loss_bpr = torch.tensor(0.0, device=DEVICE, requires_grad=True) # 或者跳過這個batch
    # else:
    #     final_pos_scores = torch.cat(expanded_pos_scores) # (total_selected_negs)
    #     final_neg_scores = torch.cat(all_selected_neg_scores) # (total_selected_negs)
    #     loss_bpr = bpr_criterion(final_pos_scores, final_neg_scores)


    # === 對於 BCE Loss ===
    # optimizer.zero_grad()
    # positive_preds = model(users, pos_items) # (batch_size)
    # positive_labels = torch.ones_like(positive_preds)
    # loss_bce_pos = bce_criterion(positive_preds, positive_labels) # 計算正樣本損失

    # all_negative_preds = []
    # for i in range(batch_size):
    #     num_selected_negs = len(selected_neg_items_list[i])
    #     if num_selected_negs == 0: continue
            
    #     user_tensor_expanded_loss = users[i].unsqueeze(0).expand(num_selected_negs)
    #     current_neg_preds = model(user_tensor_expanded_loss, selected_neg_items_list[i])
    #     all_negative_preds.append(current_neg_preds)

    # if not all_negative_preds:
    #     loss_bce_neg = torch.tensor(0.0, device=DEVICE)
    # else:
    #     final_negative_preds = torch.cat(all_negative_preds)
    #     negative_labels = torch.zeros_like(final_negative_preds)
    #     loss_bce_neg = bce_criterion(final_negative_preds, negative_labels)

    # total_loss_bce = (loss_bce_pos * batch_size + loss_bce_neg * final_negative_preds.size(0)) / (batch_size + final_negative_preds.size(0)) # 平均
    # # 或者簡單相加，或者根據正負樣本比例加權
    # # total_loss_bce = loss_bce_pos + loss_bce_neg (如果 bce_criterion 內部做了 mean)
    # # 假設 bce_criterion 已經做了 mean，那麼更準確的可能是分別計算後再平均
    # # num_pos_samples = batch_size
    # # num_neg_samples = final_negative_preds.size(0) if final_negative_preds is not None else 0
    # # if num_pos_samples + num_neg_samples > 0:
    # #     total_loss_bce = (loss_bce_pos * num_pos_samples + loss_bce_neg * num_neg_samples) / (num_pos_samples + num_neg_samples)
    # # else:
    # #     total_loss_bce = torch.tensor(0.0, device=DEVICE, requires_grad=True)


    # # loss.backward()
    # # optimizer.step()
    ```
- view_vectorized:
    ```python
        # --- 在您的訓練函數 (例如 train_bpr_model) 內部 ---
    # users, pos_items = batch # (batch_size), (batch_size)
    # batch_size = users.size(0)
    # optimizer.zero_grad()

    # 1. 準備候選負樣本 (向量化)
    #    每個正樣本對應 M 個候選負樣本
    #    neg_candidates_batch = [[] for _ in range(batch_size)] # (batch_size, M)
    #    # ... (省略了填充 neg_candidates_batch 的詳細邏輯，確保不包含 user_known_positives)
    #    # 假設 neg_candidates_batch_tensor 是一個 [batch_size, NUM_CANDIDATE_NEG_SAMPLES] 的 tensor
    #    # 這裡的採樣邏輯需要仔細設計以保持高效和正確性

    # 示例：為 batch 中的每個 user 隨機採樣 M 個全局負樣本 (不考慮個體已知正例，簡化版)
    # 這是非常簡化的版本，實際應過濾掉已知正樣本
    neg_items_indices = torch.randint(0, num_items, (batch_size, NUM_CANDIDATE_NEG_SAMPLES), device=DEVICE)

    # 2. 計算候選負樣本的得分 (向量化)
    users_expanded = users.unsqueeze(1).expand(-1, NUM_CANDIDATE_NEG_SAMPLES) # (B, M)
    # model.eval() # 可選
    with torch.no_grad(): # 不計算負樣本評分的梯度
        candidate_neg_scores = model(
            users_expanded.reshape(-1), 
            neg_items_indices.reshape(-1)
        ).view(batch_size, NUM_CANDIDATE_NEG_SAMPLES) # (B, M)
    # model.train()

    # 3. 選擇 Hard 和 Random Negatives (向量化)
    actual_k_hard = min(NUM_HARD_NEG_SAMPLES, NUM_CANDIDATE_NEG_SAMPLES)
    _, hard_indices_in_candidates = torch.topk(candidate_neg_scores, k=actual_k_hard, dim=1) # (B, N_hard)
    hard_neg_items = torch.gather(neg_items_indices, 1, hard_indices_in_candidates) # (B, N_hard)

    # 隨機選擇 (簡化：從原始候選中隨機選，可能與hard重疊，更優做法是從非hard中選)
    if NUM_RANDOM_NEG_SAMPLES > 0:
        random_selection_indices = torch.randint(0, NUM_CANDIDATE_NEG_SAMPLES, (batch_size, NUM_RANDOM_NEG_SAMPLES), device=DEVICE)
        random_neg_items = torch.gather(neg_items_indices, 1, random_selection_indices) # (B, N_random)
        final_selected_neg_items = torch.cat([hard_neg_items, random_neg_items], dim=1) # (B, N_hard + N_random)
    else:
        final_selected_neg_items = hard_neg_items # (B, N_hard)

    num_selected_negs_per_pos = final_selected_neg_items.size(1)

    # 4. 計算損失 (BPR為例)
    pos_scores = model(users, pos_items) # (B)
    pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, num_selected_negs_per_pos) # (B, N_hard + N_random)

    users_final_expanded = users.unsqueeze(1).expand(-1, num_selected_negs_per_pos) # (B, N_hard+N_random)

    selected_neg_scores = model(
        users_final_expanded.reshape(-1),
        final_selected_neg_items.reshape(-1)
    ).view(batch_size, num_selected_negs_per_pos) # (B, N_hard+N_random)

    loss = bpr_criterion(pos_scores_expanded.reshape(-1), selected_neg_scores.reshape(-1))
    # loss.backward()
    # optimizer.step()
    ```