import pandas as pd
import numpy as np

def engineer_features(df):
    """Engineers digital intent metrics and normalizes income."""
    df_out = df.copy()
    df_out.columns = [col.lower() for col in df_out.columns]
    
    if 'websitevisits' in df_out.columns and 'timeonsite' in df_out.columns:
        df_out['avg_time_per_visit'] = np.where(
            df_out['websitevisits'] > 0, 
            df_out['timeonsite'] / df_out['websitevisits'], 
            0
        )
        df_out['engagement_score'] = df_out['websitevisits'] * df_out['avg_time_per_visit']
    
    if 'income' in df_out.columns:
        df_out['log_income'] = np.log1p(df_out['income'])
        
    return df_out
