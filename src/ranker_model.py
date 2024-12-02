import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from eval_metrics import NDCG, precision_simple

class SongRecommenderXGB:
    def __init__(self, params=None):
        """Initialize the XGBoost model with optional hyperparameters."""
        self.params = params if params else {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 10,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42
        }
        self.model = XGBClassifier(**self.params)
    
    def train(self, X_train, y_train):
        """Train the model with the given training data."""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return predicted probabilities."""
        y_pred = self.model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
        return y_pred

def evaluate_metrics(y_test, y_pred):
    """Evaluate the model using AUC and other metrics."""
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score: {auc_score:.4f}")
    return auc_score

if __name__ == "__main__":
    # Step 1: Load Data
    data_path = "train_data/xgboost_train.csv"
    df = pd.read_csv(data_path)

    # Step 2: Feature Engineering
    # Create binary label: 1 if true_pos >= 0, else 0
    df['label'] = df['true_pos'].apply(lambda x: 1 if x >= 0 else 0)

    # Separate features and labels
    X = df.drop(columns=['track_id', 'true_pos', 'label'])
    y = df['label']

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 4: Train the Model
    recommender = SongRecommenderXGB()
    recommender.train(X_train, y_train)

    # Step 5: Evaluate the Model
    y_pred = recommender.evaluate(X_test, y_test)

    # Step 6: Compute Metrics
    evaluate_metrics(y_test, y_pred)

    # Optional: Rank Tracks
    df_test = X_test.copy()
    df_test['score'] = y_pred
    df_test['label'] = y_test
    df_test_sorted = df_test.sort_values(by='score', ascending=False)
    
    print("\nTop-ranked tracks:")
    print(df_test_sorted.head(10))  # Top-ranked tracks

    print("\nBottom-ranked tracks:")
    print(df_test_sorted.tail(10))  # Bottom-ranked tracks

    # Group predictions and ground truth by 'pid'
    preds = df_test.groupby('pid').apply(
    lambda group: {'pid': group.name, 'scores': group['score'].tolist()}).tolist()

    ground_truth = df_test.groupby('pid').apply(
    lambda group: {'pid': group.name, 'labels': group['label'].tolist()}).tolist()

    ndcg_score = NDCG(preds, ground_truth, k=10)  # Calculate NDCG@10
    print(f"NDCG@10 Score: {ndcg_score:.4f}")
    print(precision_simple(preds, ground_truth))
