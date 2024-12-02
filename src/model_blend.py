from collections import defaultdict
import csv
import random
import os
import sys

from scipy.sparse import coo_matrix

import latent_factor_model
import sparse_repr
import utils

random.seed(414)

def get_feature_vec(candidate, score, pid, plist_vector, true_pos, nm):
    if true_pos is None:
        true_pos = -1
    return {
        "track_id": candidate.track_id,
        "artist_id": candidate.artist_id,
        "latent_factor_score": score,
        "user_user_score": nm.user_to_user_score(plist_vector, candidate.track_id),
        "item_item_score": nm.item_to_item_score(plist_vector, candidate.track_id),
        "pid": candidate.pid,
        "predicted_pos": candidate.pos,
        "true_pos": true_pos,
    }

DATA_DIR = "validation_data"

def create_xgboost_training_data():
    lf = latent_factor_model.LatentFactors()
    nm = sparse_repr.NeighborModels()
    train_slices = [x for x in os.listdir(DATA_DIR) if x.startswith("mpd.slice")]
    xgboost_train_data = []
    slice_num = 0
    playlist_num = 0
    sampled_train_slices = random.sample(train_slices, 1)
    for filename in sampled_train_slices:
        print(f"processing slice num {slice_num}")
        slice_num += 1
        curr_slice = utils.read_track_csv(os.path.join(DATA_DIR, filename))
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
            sampled_incorrect = random.sample(incorrect_idxs, min(20, len(incorrect_idxs)))
            # NOTE (mfisher): this construction seems inefficient
            plist_vector = coo_matrix(
                (
                    [1] * len(true_tracks),
                    ([0] * len(true_tracks), [x.track_id for x in true_tracks]),
                ),
                shape=(1, nm.ns_mat.shape[1])
            )
            full_sample = [
                    get_feature_vec(
                        candidates[i],
                        scores[i],
                        pid,
                        plist_vector,
                        true_pos=true_track_pos.get(candidates[i].track_id),
                        nm=nm,
                    )
                    for i in sampled_correct + sampled_incorrect
            ]
            xgboost_train_data.extend(full_sample)
    print(len(xgboost_train_data))
    write_xgboost_csv(xgboost_train_data)

def create_xgboost_validation_data(data_dir):
    lf = latent_factor_model.LatentFactors()
    nm = sparse_repr.NeighborModels()
    slices = [x for x in os.listdir(data_dir) if x.startswith("mpd.slice")]
    xgboost_validate_data = []
    slice_num = 0
    playlist_num = 0
    for filename in slices:
        if playlist_num > 5: # temp
            break
        print(f"processing slice num {slice_num}")
        slice_num += 1
        curr_slice = utils.read_track_csv(os.path.join(data_dir, filename))
        pid_to_tracks = defaultdict(list)
        for track in curr_slice:
            pid_to_tracks[track.pid].append(track)
        for pid, true_tracks in pid_to_tracks.items():
            playlist_num += 1
            print(f"{playlist_num=}")
            if playlist_num > 5: # temp
                break
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
            # NOTE (mfisher): this construction seems inefficient
            plist_vector = coo_matrix(
                (
                    [1] * len(true_tracks),
                    ([0] * len(true_tracks), [x.track_id for x in true_tracks]),
                ),
                shape=(1, nm.ns_mat.shape[1])
            )
            full_sample = [
                    get_feature_vec(
                        candidates[i],
                        scores[i],
                        pid,
                        plist_vector,
                        true_pos=true_track_pos.get(candidates[i].track_id),
                        nm=nm,
                    )
                    for i in correct_idxs + incorrect_idxs
            ]
            xgboost_validate_data.extend(full_sample)
    print(len(xgboost_validate_data))
    write_xgboost_csv(xgboost_validate_data, data_dir)

def write_xgboost_csv(data, data_dir):
    with open(os.path.join(data_dir, "xgboost.csv"), mode='w+', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)



if __name__ == "__main__":
    data_dir = sys.argv[1]
    #create_xgboost_training_data()
    create_xgboost_validation_data(data_dir)
