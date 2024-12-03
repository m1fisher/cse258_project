from concurrent.futures import ThreadPoolExecutor

from collections import defaultdict
import csv
import random
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix

import latent_factor_model
import sparse_repr
import utils

random.seed(414)

# TODO: cache this in a less hacky way
PLIST_COMPAT_SCORE_MEANS = {}
PLIST_COMPAT_SCORE_SDS = {}
def get_feature_vec(candidate, score, pid, plist_vector, true_pos, nm, lf):
    if true_pos is None:
        true_pos = -1
    song_compat_score_mean = (lf.model_playlist.item_factors[candidate.track_id]
                              @ lf.model_playlist.item_factors[plist_vector.col].T).mean(axis=0)
    plist_compat_score_mean = PLIST_COMPAT_SCORE_MEANS.get(pid)
    plist_compat_score_sd = PLIST_COMPAT_SCORE_SDS.get(pid)
    if plist_compat_score_mean is None:
        plist_factor_vector = lf.model_playlist.item_factors[plist_vector.col].mean(axis=0)
        plist_dot_products = (plist_factor_vector
                              @ lf.model_playlist.item_factors[plist_vector.col].T)
        plist_compat_score_mean = plist_dot_products.mean(axis=0)
        plist_compat_score_sd = np.std(plist_dot_products)
        PLIST_COMPAT_SCORE_MEANS[pid] = plist_compat_score_mean
        PLIST_COMPAT_SCORE_SDS[pid] = plist_compat_score_sd
    return {
        "track_id": candidate.track_id,
        "artist_id": candidate.artist_id,
        "album_id": candidate.album_id,
        "latent_factor_score": score,
        "song_compat_score_mean": song_compat_score_mean,
        "plist_compat_score_mean": plist_compat_score_mean,
        "plist_compat_score_sd": plist_compat_score_sd,
        "pid": candidate.pid,
        "predicted_pos": candidate.pos,
        "true_pos": true_pos,
    }



def create_xgboost_training_data(data_dir):
    lf = latent_factor_model.LatentFactors()
    nm = sparse_repr.NeighborModels()
    train_slices = [x for x in os.listdir(data_dir) if x.startswith("mpd.slice")]
    slice_num = 0
    playlist_num = 0
    sampled_train_slices = sorted(train_slices)
    total_rows = 0
    for filename in sampled_train_slices:
        xgboost_train_data = []
        print(f"processing slice {filename}")
        slice_num += 1
        curr_slice = utils.read_track_csv(os.path.join(data_dir, filename))
        pid_set = set(t.pid for t in curr_slice)
        sampled_pids = random.sample(list(pid_set), len(pid_set) // 10)
        pid_to_tracks = defaultdict(list)
        for track in curr_slice:
            if track.pid in sampled_pids:
                pid_to_tracks[track.pid].append(track)
        for pid, true_tracks in pid_to_tracks.items():
            playlist_num += 1
            if playlist_num % 10 == 0:
                print(f"{playlist_num=}")
            true_track_ids = set(x.track_id for x in true_tracks)
            true_track_pos = {x.track_id: x.pos for x in true_tracks}
            candidates, scores = lf.predict_with_scores({pid: true_tracks})
            candidates = candidates[pid]
            scores = scores[pid]
            correct_idxs = [
                i for i, x in enumerate(candidates) if x.track_id in true_track_ids
            ]
            incorrect_idxs = [
                i for i, x in enumerate(candidates) if x.track_id not in true_track_ids
            ]
            # sample size 20 in http://www.cs.utoronto.ca/~mvolkovs/recsys2018_challenge.pdf
            sampled_correct = random.sample(correct_idxs, min(20, len(correct_idxs)))
            sampled_incorrect = random.sample(
                incorrect_idxs, min(20, len(incorrect_idxs))
            )
            plist_vector = coo_matrix(
                (
                    [1] * len(true_tracks),
                    ([0] * len(true_tracks), [x.track_id for x in true_tracks]),
                ),
                shape=(1, nm.ns_mat.shape[1]),
            )
            with ThreadPoolExecutor() as executor:
                full_sample = list(
                    executor.map(
                        lambda x: get_feature_vec(*x),
                        [
                            (
                                candidates[i],
                                scores[i],
                                pid,
                                plist_vector,
                                true_track_pos.get(candidates[i].track_id),
                                nm,
                                lf,
                            )
                            for i in sampled_correct + sampled_incorrect
                        ],
                    )
                )
            xgboost_train_data.extend(full_sample)
        total_rows += len(xgboost_train_data)
        print(f"processed {total_rows} rows")
        write_xgboost_csv(xgboost_train_data, data_dir)


def write_xgboost_csv(data, data_dir, mode="a"):
    fname = "xgboost_train_data.csv"
    is_new = (not os.path.exists(os.path.join(data_dir, fname)))
    with open(os.path.join(data_dir, fname), mode=mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        if is_new:
            writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    create_xgboost_training_data(data_dir)
    # create_xgboost_validation_data(data_dir)
