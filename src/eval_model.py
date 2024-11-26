"""
Given a model, evaluate the accuracy of its playlist predictions.
"""
from collections import defaultdict
import statistics

import eval_metrics
from utils import read_track_csv

### Import your model here as model ###
from popularity_baseline import model

def evaluate(predict_func):
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
    predictions = predict_func(eval_per_playlist)

    r_precisions = []
    clicks = []
    for pid in ground_truth_per_playlist:
        r_precisions.append(eval_metrics.R_precision(predictions[pid], ground_truth_per_playlist[pid]))
        clicks.append(eval_metrics.clicks(predictions[pid], ground_truth_per_playlist[pid]))
    print(f"Average R-precision: {statistics.mean(r_precisions)}")
    print(f"Average num clicks: {statistics.mean(clicks)}")





if __name__ == "__main__":
    evaluate(model)
