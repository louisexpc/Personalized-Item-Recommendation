# Personalized-Item-Recommendation

# Project Overview

```
R13945035/ 
├── main.py # entry point
├── model.py
├── train.py
├── preprocessing.py
├── config.py
├── requirements.txt
├── run.sh
├── README.md
├── config/
		│ ├── data.yaml
		│ ├── train_bpr.yaml
		│ ├── train_bce.yaml 

```

### Config 檔案與參數詳解

存放資料前處理與模型訓練之參數檔

必要檔案包含：

- `data.yaml` : 資料前處理參數檔，內涵參數
    - `split_mode`:
        - `"fixed"`  : 每個 User 按照固定數量 `val_n` 隨機選取作為 validation
        ❗數量不足 `val_n` 的 User 會全數做完 training data ，在 validation 時會跳過
        - `"ratio"` : 每個 User 按照固定比例 `val_ratio` 隨機選取作為 validation
    - `neg_mode`:
        - `"fixed"`：固定為每個 User 產生 `neg_sample_num` 個負樣本：
        - `"even"`： 按每個 User 的正樣本數量 1 : 1 採樣
    - `val_ratio`:  搭配 `split_mode` ”ratio” 使用
    - `val_n` : 搭配 `split_mode` ”val_n” 使用
    - `neg_sample_num`: 搭配 `neg_mode` ”fixed” 使用
- `train_bpr.yaml` : BPR Loss 訓練參數
    - `device` : 決定是否使用 CPU 或 GPU 訓練
    - `latent_factors`  : MF Model 的  hidden factors 數量
    - `epochs`, `lr`, `batch_size`, `weight_decay` : Model 訓練參數
    - `top_k` : 計算 `mAP@` 的數量
    - `margin` : BPR Loss 參數
    - `num_candidate_neg_samples` : Hard Negative Sampling 參數，初步抽樣的候選負樣本數 (N)
    - `num_hard_neg_samples`  : Hard Negative Sampling 參數，從候選中篩選出的 top-k 困難負樣本數 (M)
    - `num_random_neg_samples`  :Hard Negative Sampling 參數， 額外加入的隨機負樣本數(K)
- `train_bce.yaml` : BCE Loss 訓練參數
    - `device` : 決定是否使用 CPU 或 GPU 訓練
    - `latent_factors`  : MF Model 的  hidden factors 數量
    - `epochs`, `lr`, `batch_size`, `weight_decay` : Model 訓練參數
    - `top_k` : 計算 `mAP@` 的數量

## Quick Start

```bash
pip install -r requirements.txt
chmod +x ./run.sh
./run.sh -o ./
```

### 支援參數說明

- `-o` : `required`, 輸出 submission.csv 的儲存路徑
- `-bce` : `store ture`，改啟用 BCE Loss 訓練模型，若未指定參數則預設使用 BPR Loss