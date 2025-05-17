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
    parser.add_argument("-bce",action="store_true", help="Using WF with BCE Loss")

    args = parser.parse_args()

    if not os.path.exists(os.path.join("train.csv")):
        raise ValueError(f"train.csv not found!")
    
    train_data_path = os.path.join("train.csv")

    if not os.path.exists(os.path.join("config","data.yaml")):
        raise ValueError(f"data.yaml not found!")
    data_config = load_config(os.path.join("config","data.yaml"))

    if args.bce:
        print("Using WF with BCE Loss, Loading train_bce.yaml")
        if not os.path.exists(os.path.join("config","train_bce.yaml")):
            raise ValueError(f"train_bce.yaml not found!")
        train_config = load_config(os.path.join("config","train_bce.yaml"))
        
    else:
        print("Using WF with BPR Loss: Loading train_bpr.yaml")
        if not os.path.exists(os.path.join("config","train_bpr.yaml")):
            raise ValueError(f"train_bpr.yaml not found!")
        train_config = load_config(os.path.join("config","train_bpr.yaml"))
    
    device = torch.device(train_config.device)
    print(f"Using device: {device}")

    """Preprocessing Data"""
    bce_train_data,bpr_train_data, \
    idx_to_user,idx_to_item , \
    num_items, num_users , \
    train_pos_dict,val_pos_dict, train_mapping = load_and_preprocessing(train_data_path, data_config)

    if args.bce:
        model_bce = LFM(num_users, num_items, train_config.latent_factors).to(device)

        trained_model_bce = train_bce_model(
            model_bce, 
            bce_train_data, 
            train_pos_dict,
            val_pos_dict,
            num_items,
            train_config,
        )

        best_model_bce = load_model(
            model_path="best_bce_model.pth",
            num_users=num_users,
            num_items=num_items,
            num_factors= train_config.latent_factors,
            device=device
        )

        final_predictions_df = predict_top_k_for_all_users(
            model = best_model_bce,
            train_mapping = train_mapping, 
            num_items = num_items, 
            num_users = num_users,
            idx_to_user = idx_to_user,
            idx_to_item = idx_to_item,
            device = device, 
            top_k = 50
        )
        
        
       
        output_filename = "submission_bce.csv"
        final_predictions_df.to_csv(output_filename, index=False)
        print(f"\nPredictions saved to {output_filename}")
    else:
        model_bpr = LFM(num_users, num_items, train_config.latent_factors).to(device)
        """Random Negative Sampling"""
        # trained_model_bpr = train_bpr_model(
        #     model=model_bpr,
        #     train_data=bpr_train_data,
        #     train_pos_dict=train_pos_dict,
        #     val_pos_dict=val_pos_dict,
        #     num_items=num_items,
        #     train_config=train_config,
        # )
        """Hard Negative Sampling"""
        trained_model_bpr = train_bpr_model_with_hard_negative_sampling_vectorize(
            model=model_bpr,
            train_mapping= train_mapping,
            train_pos_dict=train_pos_dict,
            val_pos_dict=val_pos_dict,
            num_items=num_items,
            train_config=train_config,
        )

        
        best_model_bpr = load_model(
            model_path="best_bpr_model.pth",
            num_users=num_users,
            num_items=num_items,
            num_factors= train_config.latent_factors,
            device=device
        )

        final_predictions_df = predict_top_k_for_all_users(
            model = best_model_bpr,
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
    
    end_overall_time = time.time()
    total_runtime = end_overall_time - start_overall_time
    print(f"Total script runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")

    if total_runtime > 300: 
        print("WARNING: Total runtime exceeded 5 minutes!")




def load_model(
    model_path: str,
    num_users: int, 
    num_items:int, 
    num_factors:int,
    device: str = "cpu",
    ) -> torch.nn.Module:

    model = LFM(num_users,num_items,num_factors)
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    
    model.to(device)
    model.eval()
    return model
        

if __name__=="__main__":
    main()