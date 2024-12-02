import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from eval_metrics import NDCG, precision_simple

import utils

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

from scipy.sparse import coo_matrix
import latent_factor_model
import sparse_repr
from model_blend import get_feature_vec

import pickle
# will move into class
lf = latent_factor_model.LatentFactors()
nm = sparse_repr.NeighborModels()

def predict(playlists):
    # Create X dataframe from playlists
    X_data = []
    playlist_num = 0
    for pid, seed_tracks in playlists.items():
        print(f"{playlist_num=}")
        playlist_num += 1
        candidates, scores = lf.predict_with_scores({pid: seed_tracks})
        candidates = candidates[pid]
        scores = scores[pid]
        plist_vector = coo_matrix(
            (
                [1] * len(seed_tracks),
                ([0] * len(seed_tracks), [x.track_id for x in seed_tracks]),
            ),
            shape=(1, nm.ns_mat.shape[1])
        )
        X_data.extend([
            get_feature_vec(
                candidates[i],
                scores[i],
                pid,
                plist_vector,
                nm=nm,
                true_pos=-1,
            ) for i in range(len(candidates))
        ])
    X = pd.DataFrame(X_data)
    X = X.drop(columns=["true_pos"])
    # TODO: formalize /move this into class
    #recommender = pickle.load(open("xgboost_model.pkl", "rb"))
    recommender = SongRecommenderXGB()
    recommender.model.load_model("xgb_model")

    # Predict given candidate track features
    preds = [p[1] for p in recommender.model.predict_proba(X)]
    X['score'] = preds
    X_pids = X.groupby('pid').apply(
        lambda group: {'pid': group.name,
                       'scores': group['score'].tolist(),
                       'track_id': group['track_id'].to_list(),
                       'artist_id': group['artist_id'].to_list()}).tolist()
    final_preds = {}
    for pred in X_pids:
        score_idxs = [(score, i) for i, score in enumerate(pred['scores'])]
        score_idxs.sort(reverse=True)
        track_preds = []
        pos_idx = 0
        for score, i in score_idxs:
            track_preds.append(utils.Track(
                pid=pred['pid'],
                pos=pos_idx,
                track_id=pred['track_id'][i],
                artist_id=pred['artist_id'][i],
                album_id=None
            ))
            pos_idx += 1
        final_preds[pred['pid']] = track_preds
    return final_preds

if __name__ == "__main__":
    # Step 1: Load Data
    data_path = "train_data/xgboost_train.csv"
    df = pd.read_csv(data_path)

    # Step 2: Feature Engineering
    # Create binary label: 1 if true_pos >= 0, else 0
    df['label'] = df['true_pos'].apply(lambda x: 1 if x >= 0 else 0)

    # Separate features and labels
    X = df.drop(columns=['true_pos', 'label'])
    y = df['label']

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 4: Train the Model
    recommender = SongRecommenderXGB()
    recommender.train(X_train, y_train)
    recommender.model.save_model("xgb_model")

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
