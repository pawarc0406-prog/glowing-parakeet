import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_and_score(df, feature_cols, target_col):
    """Trains XGBoost propensity model and assigns scores and deciles."""
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.03,
        n_estimators=60,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    df_scored = df.copy()
    df_scored['propensity_score'] = model.predict_proba(X)[:, 1]
    
    return model, df_scored, test_auc
