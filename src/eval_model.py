"""
Given a model, evaluate the accuracy of its playlist predictions.
"""
from collections import defaultdict
import os
import random
import statistics
import sys

import eval_metrics
from utils import read_track_csv

### Import your model here as model ###
from popularity_baseline import predict
#from sparse_repr import inner_product_predict as predict
#from voyager_model import predict as model
from latent_factor_model import LatentFactors
from ranker_model import predict

random.seed(4141414)


def evaluate_all(predict_func, data_dir, model_name, quick_mode=False):
    filenames = [
        "title_and_first_1_tracks.csv",
        "title_and_first_5_tracks.csv",
        "first_5_tracks.csv",
        "first_10_tracks.csv",
        "title_and_first_10_tracks.csv",
        "title_and_first_25_tracks.csv",
        "title_and_random_25_tracks.csv",
        "title_and_first_100_tracks.csv",
        "title_and_random_100_tracks.csv",
    ]
    for fn in filenames:
        evaluate(predict_func, data_dir, fn, model_name, quick_mode)

def evaluate(predict_func, data_dir, filename, model_name, quick_mode=False):
    evaluation_file = os.path.join(data_dir, filename)
    eval_playlists = read_track_csv(evaluation_file)
    eval_per_playlist = defaultdict(list)
    for track in eval_playlists:
        eval_per_playlist[track.pid].append(track)
    ground_truth_file = f"{data_dir}/ground_truth_{filename}"
    ground_truth = read_track_csv(ground_truth_file)
    ground_truth_per_playlist = defaultdict(list)
    for track in ground_truth:
        ground_truth_per_playlist[track.pid].append(track)
    if quick_mode == True:
        n = 5
        rand_idxs = set(random.sample(range(len(eval_per_playlist)), n))
        print(rand_idxs)
        eval_per_playlist = {k: v for i, (k,v) in enumerate(list(eval_per_playlist.items())) if i in rand_idxs}
        ground_truth_per_playlist = {k: v for i, (k,v) in enumerate(list(ground_truth_per_playlist.items())) if i in rand_idxs}
    predictions = predict_func(eval_per_playlist)

    recalls = []
    r_precisions = []
    clicks = []
    ndcg = []
    for pid in ground_truth_per_playlist:
        recalls.append(eval_metrics.recall(predictions[pid], ground_truth_per_playlist[pid]))
        r_precisions.append(eval_metrics.R_precision(predictions[pid], ground_truth_per_playlist[pid]))
        clicks.append(eval_metrics.clicks(predictions[pid], ground_truth_per_playlist[pid]))
        ndcg.append(eval_metrics.NDCG(predictions[pid], ground_truth_per_playlist[pid]))
    print(filename)
    print(f"Average recall: {statistics.mean(recalls)}")
    print(f"Average R-precision: {statistics.mean(r_precisions)}")
    print(f"Average num clicks: {statistics.mean(clicks)}")
    print(f"Average NDCG: {statistics.mean(ndcg)}")
    with open(os.path.join(data_dir, f"{model_name}_eval_stats.txt"), "a") as fh:
        fh.write(filename + "\n")
        fh.write(f"Average recall: {statistics.mean(recalls)}\n")
        fh.write(f"Average R-precision: {statistics.mean(r_precisions)}\n")
        fh.write(f"Average num clicks: {statistics.mean(clicks)}\n")
        fh.write(f"Average NDCG: {statistics.mean(ndcg)}\n")


if __name__ == "__main__":
    data_dir = sys.argv[1]
    model_name = "LatentFactors"
    quick_mode = False
    if len(sys.argv) > 2 and sys.argv[2] == "--quick":
        quick_mode = True
    evaluate_all(LatentFactors().predict, data_dir, model_name, quick_mode)
    #evaluate(LatentFactors().predict, quick_mode)
