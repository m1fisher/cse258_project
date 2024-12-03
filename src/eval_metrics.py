import numpy as np
import statistics

def R_precision(preds: list[dict], ground_truth: list[dict]):
    """
    Expected entry for preds or ground_truth:
    {"track": x, "artist": y}
    """
    G_t = set(x.track_id for x in ground_truth)
    G_a = set(x.artist_id for x in ground_truth)
    preds = preds[:len(ground_truth)]
    S_t = set(x.track_id for x in preds)
    S_a = set(x.artist_id for x in preds)
    numerator = len(S_t.intersection(G_t)) + 0.25 * len(S_a.intersection(G_a))
    return numerator / len(G_t)

def precision_simple(preds, ground_truth):
    """
    simple precision function for xgboost model
    # TODO: Fix this to include artist_id for correct Spotify R-precision
    """
    precision_scores = []
    for pred, truth in zip(preds, ground_truth):
        assert pred['pid'] == truth['pid'], "Playlist IDs must match between predictions and ground truth."
        # Extract scores and labels
        scores = pred['scores']
        labels = truth['labels']
        # Sort items by predicted scores in descending order
        sorted_items = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        sorted_labels = [item[1] for item in sorted_items]
        if len(sorted_items) <= 1:
            print("too short, skipping")
            continue

        num_correct = sum(sorted_labels[i] == 1 for i in range(len(sorted_items) // 2))
        precision_scores.append(num_correct / (len(sorted_items) // 2))
    return statistics.mean(precision_scores)


def recall(preds: list[dict], ground_truth: list[dict]):
    print(len(preds))
    G_t = set(x.track_id for x in ground_truth)
    S_t = set(x.track_id for x in preds)
    return len(S_t.intersection(G_t)) / len(G_t)

def DCG_xgboost(scores, labels, k=None):
    """
    Compute Discounted Cumulative Gain (DCG) for a list of scores and labels.
    Args:
        scores (list[float]): Predicted scores for the items.
        labels (list[int]): Ground truth relevance labels (0 or 1).
        k (int): The rank position up to which to calculate DCG. Defaults to full list.
    Returns:
        float: The DCG score.
    """
    # Sort items by predicted scores in descending order
    sorted_items = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    sorted_labels = [item[1] for item in sorted_items]

    # Compute DCG
    if k is None:
        k = len(sorted_labels)
    dcg = sum(
        (2**rel - 1) / np.log2(idx + 2)  # log2(idx+2) because idx is 0-based
        for idx, rel in enumerate(sorted_labels[:k])
    )
    return dcg

def DCG(preds, ground_truth):
    """
    Compute Discounted Cumulative Gain (DCG) for a list of scores and labels.
    Returns:
        float: The DCG score.
    """
    #sorted_items = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    #sorted_labels = [item[1] for item in sorted_items]
    true_idx = {t.track_id: i for i, t in enumerate(ground_truth)}

    # Compute DCG
    dcg = sum(
        true_idx.get(track.track_id, 0) / np.log2(idx + 2)  # log2(idx+2) because idx is 0-based
        for idx, track in enumerate(preds)
    )
    return dcg


def NDCG(preds, ground_truth):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    Args:
    Returns:
        float: The average NDCG score across all playlists.
    """
    dcg = DCG(preds, ground_truth)
    idcg = DCG(ground_truth, ground_truth)

    # Avoid division by zero if IDCG is 0
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


def NDCG_xgboost(preds, ground_truth, k=None):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    Args:
        preds (list[dict]): Predicted rankings for each playlist. Each dict should have:
                            {'pid': playlist_id, 'scores': [score1, score2, ...]}.
        ground_truth (list[dict]): Ground truth relevance for each playlist. Each dict should have:
                                    {'pid': playlist_id, 'labels': [label1, label2, ...]}.
        k (int): The rank position up to which to calculate NDCG. Defaults to full list.
    Returns:
        float: The average NDCG score across all playlists.
    """
    ndcg_scores = []
    for pred, truth in zip(preds, ground_truth):
        assert pred['pid'] == truth['pid'], "Playlist IDs must match between predictions and ground truth."

        # Extract scores and labels
        scores = pred['scores']
        labels = truth['labels']

        # Compute DCG and ideal DCG (IDCG)
        dcg = DCG_xgboost(scores, labels, k)
        idcg = DCG_xgboost(labels, labels, k)  # Sort by actual labels for ideal ranking

        # Avoid division by zero if IDCG is 0
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    # Return the mean NDCG across all playlists
    return np.mean(ndcg_scores)

def clicks(preds: list[dict], ground_truth: list[dict]):
    """
    Number of clicks necessary to reach a correct track.
    """
    ground_tracks = set(x.track_id for x in ground_truth)
    good_idx = 511
    for i, pred in enumerate(preds):
        if pred.track_id in ground_tracks:
            good_idx = i
            break
    # Note (mfisher): I've ommitted a -1 from good_idx here
    # compared to the Spotify paper; I think theirs is 1-indexed.
    # The idea is that, if good_idx is in top 10, clicks value should be 0.
    return good_idx // 10
