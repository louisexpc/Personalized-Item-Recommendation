import time
import os
import torch
import argparse

from config import load_config, Config
from model import LFM

from preprocessing import load_and_preprocessing
from train import train_bpr_model, train_bce_model, predict_top_k_for_all_users, train_bpr_model_with_hard_negative_sampling, train_bpr_model_with_hard_negative_sampling_vectorize

def main():
    start_overall_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-o",required=True,help="Output ranked list file")
    parser.add_argument("-d",required=True,help="path of training data")
    parser.add_argument("-bpr",action="store_true", help="Using BPR model")

    args = parser.parse_args()

    if not os.path.exists(args.d):
        raise ValueError(f"train.csv not found!")
    if not os.path.exists(os.path.join("config","data.yaml")):
        raise ValueError(f"data.yaml not found!")
    
    data_config = load_config(os.path.join("config","data.yaml"))

    if args.bpr:
        print("Using BPR Model")
        if not os.path.exists(os.path.join("config","train_bpr.yaml")):
            raise ValueError(f"train_bpr.yaml not found!")
        train_config = load_config(os.path.join("config","train_bpr.yaml"))
    else:
        print("Using BCE Model")
        if not os.path.exists(os.path.join("config","train_bce.yaml")):
            raise ValueError(f"train_bce.yaml not found!")
        train_config = load_config(os.path.join("config","train_bce.yaml"))
  
    
    device = torch.device(train_config.device)
    print(f"Using device: {device}")

    bce_train_data,bpr_train_data, \
    idx_to_user,idx_to_item , \
    num_items, num_users , \
    train_pos_dict,val_pos_dict, train_mapping = load_and_preprocessing(args.d, data_config)

    if args.bpr :
        # --- 2. Model Initialization (BPR) ---
        model_bpr = LFM(num_users, num_items, train_config.latent_factors).to(device)

        # --- 3. Training (BPR) ---
        # trained_model_bpr = train_bpr_model(
        #     model=model_bpr,
        #     train_data=bpr_train_data,
        #     train_pos_dict=train_pos_dict,
        #     val_pos_dict=val_pos_dict,
        #     num_items=num_items,
        #     train_config=train_config,
        # )
        trained_model_bpr = train_bpr_model_with_hard_negative_sampling_vectorize(
            model=model_bpr,
            train_mapping= train_mapping,
            train_pos_dict=train_pos_dict,
            val_pos_dict=val_pos_dict,
            num_items=num_items,
            train_config=train_config,
        )

        # --- 4. Prediction (BPR) ---
        final_predictions_df = predict_top_k_for_all_users(
            model = trained_model_bpr,
            train_mapping = train_mapping, 
            num_items = num_items, 
            num_users = num_users,
            idx_to_user = idx_to_user,
            idx_to_item = idx_to_item,
            device = device, 
            top_k = 50
        )
        # --- 5. Generate Output CSV ---
        
       
        output_filename = "submission_bpr.csv"
        final_predictions_df.to_csv(output_filename, index=False)
        print(f"\nPredictions saved to {output_filename}")
      

    else:
        # --- 2. Model Initialization (BCE) ---
        model_bce = LFM(num_users, num_items, train_config.latent_factors).to(device)

        # --- 3. Training (BCE) ---
        trained_model_bce = train_bce_model(
            model_bce, 
            bce_train_data, 
            train_pos_dict,
            val_pos_dict,
            num_items,
            train_config,
        )
        
        # --- 4. Prediction (BCE) ---
        final_predictions_df = predict_top_k_for_all_users(
            model = trained_model_bce,
            train_mapping = train_mapping, 
            num_items = num_items, 
            num_users = num_users,
            idx_to_user = idx_to_user,
            idx_to_item = idx_to_item,
            device = device, 
            top_k = 50
        )
        # --- 5. Generate Output CSV ---
        
       
        output_filename = "submission_bce.csv"
        final_predictions_df.to_csv(output_filename, index=False)
        print(f"\nPredictions saved to {output_filename}")


    end_overall_time = time.time()
    total_runtime = end_overall_time - start_overall_time
    print(f"Total script runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")

    if total_runtime > 300: # 5 minutes
        print("WARNING: Total runtime exceeded 5 minutes!")
        

if __name__=="__main__":
    main()