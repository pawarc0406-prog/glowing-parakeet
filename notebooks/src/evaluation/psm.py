import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def evaluate_psm_lift(df, treatment_col='treatment_targeted', outcome_col='conversion'):
    """Performs 1:1 Nearest-Neighbor PSM and calculates Average Treatment Effect (ATE)."""
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[['propensity_score']])
    
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    matched_control = control.iloc[indices.flatten()].copy()
    
    treated_rate = treated[outcome_col].mean()
    control_rate = matched_control[outcome_col].mean()
    incremental_lift = treated_rate - control_rate
    
    return {
        'treated_conversion': treated_rate,
        'matched_control_conversion': control_rate,
        'incremental_lift_ate': incremental_lift
    }
