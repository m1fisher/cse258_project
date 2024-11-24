# CSE258 Assignment 2

Original challenge description [here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

Actually downloaded dataset [here](https://www.kaggle.com/datasets/himanshuwagh/spotify-million)

Useful papers:

    * An Analysis of Approaches Taken in the ACM RecSys Challenge 2018 for Automatic Music Playlist Continuation (in repo)
    * http://www.cs.utoronto.ca/~mvolkovs/recsys2018_challenge.pdf

## Cookbook

```
# First, unzip the archive such that the data is available in subdirectory `data/`
# Create test data and move it out of the data/ dir
python3 src/make_test_data.py data test --mv
# Split the test data into different evaluation categories used in the Spotify challenge
python3 src/split_test_data.py test_data/mpd.test.json
# Create validation data and move it out of the data/ dir
python3 src/make_test_data.py data validation --mv
# Split the validation data into different evaluation categories used in the Spotify challenge
python3 src/split_test_data.py validation_data/mpd.validation.json
# What remains in the `data/` subdir may now be used as training data
```

## Evaluation

Text below is taken from https://web.archive.org/web/20201202132814/https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

Submissions will be evaluated using the following metrics. All metrics will be evaluated at both the track level (exact track match) and the artist level (any track by the same artist is a match).

In the following, we denote the ground truth set of tracks by \(G\) and the ordered list of recommended tracks by \(R\). The size of a set or list is denoted by \(|\cdot| \), and we use from:to-subscripts to index a list. In the case of ties on individual metrics, earlier submissions are ranked higher.
R-precision

R-precision is the number of retrieved relevant tracks divided by the number of known relevant tracks (i.e., the number of withheld tracks):

\(\text{R-precision} = \frac{\left| G \cap R_{1:|G|} \right|}{|G|} \)

The metric is averaged across all playlists in the challenge set. This metric rewards total number of retrieved relevant tracks (regardless of order).
Normalized Discounted Cumulative Gain (NDCG)

Discounted Cumulative Gain (DCG) measures the ranking quality of the recommended tracks, increasing when relevant tracks are placed higher in the list. Normalized DCG (NDCG) is determined by calculating the DCG and dividing it by the ideal DCG in which the recommended tracks are perfectly ranked:

\(DCG = rel_1 + \sum_{i=2}^{|R|} \frac{rel_i}{\log_2 i}\)

The ideal DCG or IDCG is, in our case, equal to:

\(IDCG = 1 + \sum_{i=2}^{\left| G \cap R \right|} \frac{1}{\log_2 i} \)

If the size of the set intersection of \(G\) and \(R\) is empty, then the IDCG is equal to 0. The NDCG metric is now calculated as:

\(NDCG = \frac{DCG}{IDCG}\)
Recommended SongS Clicks

Recommended Songs is a Spotify feature that, given a set of tracks in a playlist, recommends 10 tracks to add to the playlist. The list can be refreshed to produce 10 more tracks. Recommended Songs clicks is the number of refreshes needed before a relevant track is encountered. It is calculated as follows:

\(\text{clicks} = \left\lfloor \frac{ \arg\min_i \{ R_i\colon R_i \in G|\} - 1}{10} \right\rfloor\)

If the metric does not exist (i.e. if there are no relevant tracks in \(R\), a value of 51 is picked (which is 1 greater than the maximum number of clicks possible).
Rank Aggregation

Final rankings will be computed by using the Borda Count election strategy. For each of the rankings of p participants according to R-precision, NDCG, and Recommended Songs Clicks, the top ranked system receives p points, the second system received p-1 points, and so on. The participant with the most total points wins. In the case of ties, we use top-down comparison: compare the number of 1st place positions between the systems, then 2nd place positions, and so on.
