"""
Given a model, evaluate the accuracy of its playlist predictions.
"""
from collections import defaultdict
import random
import statistics
import sys

import eval_metrics
from utils import read_track_csv

### Import your model here as model ###
#from popularity_baseline import model as predict
from sparse_repr import inner_product_predict as predict
#from voyager_model import predict as model
from latent_factor_model import LatentFactors

random.seed(414)

# TODO (mfisher): Generalize this eval pipeline across validation files
def evaluate(predict_func, quick_mode=False):
    evaluation_file = "validation_data/title_and_first_100_tracks.csv"
    eval_playlists = read_track_csv(evaluation_file)
    eval_per_playlist = defaultdict(list)
    for track in eval_playlists:
        eval_per_playlist[track.pid].append(track)
    ground_truth_file = "validation_data/ground_truth_title_and_first_100_tracks.csv"
    ground_truth = read_track_csv(ground_truth_file)
    ground_truth_per_playlist = defaultdict(list)
    for track in ground_truth:
        ground_truth_per_playlist[track.pid].append(track)
    if quick_mode == True:
        n = 500
        rand_idxs = set(random.sample(range(len(eval_per_playlist)), n))
        print(rand_idxs)
        eval_per_playlist = {k: v for i, (k,v) in enumerate(list(eval_per_playlist.items())) if i in rand_idxs}
        ground_truth_per_playlist = {k: v for i, (k,v) in enumerate(list(ground_truth_per_playlist.items())) if i in rand_idxs}
    predictions = predict_func(eval_per_playlist)

    recalls = []
    r_precisions = []
    clicks = []
    for pid in ground_truth_per_playlist:
        recalls.append(eval_metrics.recall(predictions[pid], ground_truth_per_playlist[pid]))
        r_precisions.append(eval_metrics.R_precision(predictions[pid], ground_truth_per_playlist[pid]))
        clicks.append(eval_metrics.clicks(predictions[pid], ground_truth_per_playlist[pid]))
    print(f"Average recall: {statistics.mean(recalls)}")
    print(f"Average R-precision: {statistics.mean(r_precisions)}")
    print(f"Average num clicks: {statistics.mean(clicks)}")





if __name__ == "__main__":
    quick_mode = False
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_mode = True
    evaluate(LatentFactors().predict, quick_mode)
