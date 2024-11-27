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

def recall(preds: list[dict], ground_truth: list[dict]):
    print(len(preds))
    G_t = set(x.track_id for x in ground_truth)
    S_t = set(x.track_id for x in preds)
    return len(S_t.intersection(G_t)) / len(G_t)

def NDCG(preds: list[dict], ground_truth: list[dict]):
    """
    Normalized Discounted Cumulative Gain
    """
    # TODO (mfisher): implement
    pass

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
