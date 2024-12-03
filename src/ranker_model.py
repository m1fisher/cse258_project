from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRanker
from eval_metrics import NDCG_xgboost, precision_simple

import utils


class SongRecommenderXGBRanker:
    def __init__(self, params=None):
        """Initialize the XGBRanker model with optional hyperparameters."""
        self.params = params if params else {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 10,
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg',
            'random_state': 42
        }
        self.model = XGBRanker(**self.params)

    def train(self, X_train, y_train, group):
        """Train the model with the given training data."""
        self.model.fit(X_train, y_train, group=group)

    def evaluate(self, X_test):
        """Evaluate the model and return predicted scores."""
        return self.model.predict(X_test)


def assign_qid(df, qid_column):
    """
    Assign a 'qid' (query ID) for ranking. This is based on grouping by 'qid_column' (e.g., 'pid').
    """
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df[qid_column])


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
        #with ThreadPoolExecutor(max_workers=8) as executor:
        features = list(
                map(
                    lambda x: get_feature_vec(*x), [
                        (
                            candidates[i],
                            scores[i],
                            pid,
                            plist_vector,
                            -1,  # fake true_pos
                            nm,
                            lf,
                        )
                        for i in range(len(candidates))
                    ],
                )
            )
        X_data.extend(features)
    X = pd.DataFrame(X_data)
    X["qid"] = X["pid"]
    X = X.drop(columns=["true_pos", "pid"])
    # TODO: formalize /move this into class
    #recommender = pickle.load(open("xgboost_model.pkl", "rb"))
    recommender = SongRecommenderXGBRanker()
    recommender.model.load_model("xgb_model")

    # Predict given candidate track features
    preds = recommender.model.predict(X)
    X['score'] = preds
    X_pids = X.groupby('qid').apply(
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
    data_path = "train_data/xgboost_train_data.csv"
    df = pd.read_csv(data_path)

    # Step 2: Feature Engineering
    # Create binary label: 1 if true_pos >= 0, else 0
    df['label'] = df['true_pos'].apply(lambda x: 1 if x >= 0 else 0)

    # Assign 'qid' based on 'pid'
    df['qid'] = assign_qid(df, 'pid')

    # Separate features and labels
    X = df.drop(columns=['true_pos', 'label', 'pid'])
    y = df['label']
    qid = df['qid']

    # Step 3: Train-Test Split
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    qid_train, qid_test = qid.loc[train_idx], qid.loc[test_idx]

    # Step 4: Ensure Data is Sorted by qid
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data = train_data.sort_values(by='qid')  # Sort by qid
    print(train_data.head)
    X_train_sorted = train_data.drop(columns=['label', 'qid'])
    y_train_sorted = train_data['label']
    group_train = train_data['qid'].value_counts(sort=False).sort_index().tolist()

    # Step 5: Train the Model
    recommender = SongRecommenderXGBRanker()
    recommender.train(X_train_sorted, y_train_sorted, group_train)
    recommender.model.save_model("xgb_model")

    # Step 6: Evaluate the Model
    y_pred = recommender.evaluate(X_test)

    # Step 7: Compute Metrics
    df_test = X_test.copy()
    df_test['score'] = y_pred
    df_test['label'] = y_test
    df_test['pid'] = df.loc[test_idx, 'pid']

    # Sort by 'score' for ranking
    df_test_sorted = df_test.sort_values(by='score', ascending=False)

    print("\nTop-ranked tracks:")
    print(df_test_sorted.head(10))  # Top-ranked tracks

    print("\nBottom-ranked tracks:")
    print(df_test_sorted.tail(10))  # Bottom-ranked tracks

    # Group predictions and ground truth by 'pid'
    preds = df_test_sorted.groupby('pid').apply(
        lambda group: {'pid': group.name, 'scores': group['score'].tolist()}
    ).tolist()

    ground_truth = df_test_sorted.groupby('pid').apply(
        lambda group: {'pid': group.name, 'labels': group['label'].tolist()}
    ).tolist()

    ndcg_score = NDCG_xgboost(preds, ground_truth, k=10)  # Calculate NDCG@10
    print(f"NDCG@10 Score: {ndcg_score:.4f}")

    precision_score = precision_simple(preds, ground_truth)
    print(f"Precision@10 Score: {precision_score:.4f}")
