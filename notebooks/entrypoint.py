import pandas as pd
import yaml
import os
from src.data.features import engineer_features
from src.models.propensity import train_and_score
from src.evaluation.psm import evaluate_psm_lift

def main():
    print("=== Starting FinTech Digital Acquisition Propensity Pipeline ===")
    
    # 1. Load Data
    data_path = "data/processed_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please ensure data is loaded.")
        return
        
    df_raw = pd.read_csv(data_path)
    
    # 2. Feature Engineering
    df_engineered = engineer_features(df_raw)
    
    # 3. Model Training & Propensity Scoring
    feature_cols = ['age', 'income', 'websitevisits', 'timeonsite', 'avg_time_per_visit', 'engagement_score']
    model, df_scored, test_auc = train_and_score(df_engineered, feature_cols, target_col='treatment_targeted')
    
    print(f"Propensity Model Training Complete. Test ROC-AUC: {test_auc:.4f}")
    
    # Save Scored Prospects
    df_scored.to_csv("data/scored_prospects.csv", index=False)
    
    # 4. PSM Causal Lift Evaluation
    lift_results = evaluate_psm_lift(df_scored)
    
    print("\n=== Causal Lift & Unit Economics Summary ===")
    print(f"Treated Group Conversion Rate:         {lift_results['treated_conversion']:.2%}")
    print(f"Matched Control Group Conversion Rate: {lift_results['matched_control_conversion']:.2%}")
    print(f"Incremental Conversion Lift (ATE):      {lift_results['incremental_lift_ate']:.2%}")
    print("===============================================================")

if __name__ == "__main__":
    main()
