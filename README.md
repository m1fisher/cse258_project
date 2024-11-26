# CSE258 Assignment 2

Original challenge description [here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

Actually downloaded dataset [here](https://www.kaggle.com/datasets/himanshuwagh/spotify-million)

Useful papers:

    * An Analysis of Approaches Taken in the ACM RecSys Challenge 2018 for Automatic Music Playlist Continuation (in repo)
    * http://www.cs.utoronto.ca/~mvolkovs/recsys2018_challenge.pdf

## Cookbook

```
# First, unzip the archive such that the data is available in subdirectory `data/`
# Process data to smaller CSVs
python3 src/process_data.py data processed_data
# Create test data and move it out of the processed_data/ dir
python3 src/make_test_data.py processed_data test --mv
# Split the test data into different evaluation categories used in the Spotify challenge
python3 src/split_test_data.py test_data/mpd.test.csv
# Create validation data and move it out of the data/ dir
python3 src/make_test_data.py processed_data validation --mv
# Split the validation data into different evaluation categories used in the Spotify challenge
python3 src/split_test_data.py validation_data/mpd.validation.csv
# What remains in the `processed_data/` subdir may now be used as training data
mv processed_data/ train_data/
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

Tasks

1.
2.
3.
4. Describe literature related to the problem you are studying. If you are using an existing dataset,
   where did it come from and how was it used? What other similar datasets have been studied in
   the past and how? What are the state-of-the-art methods currently employed to study this type
   of data? Are the conclusions from existing work similar to or different from your own findings?

The problem we are studying is about recommending the next song in an Automatic Playlist Continuation (APC) task. We are using an existing dataset provided by Spotify, one million user generated playlists.

It was generated for the ACM RecSys challenge 2018 for Automatic Music Playlist Continuation.

Describe literature related to the problem you are studying.

What other similar datasets have been studied in
the past and how?

What are the state-of-the-art methods currently employed to study this type
of data?

From 'Analysis of Approaches Taken in the ACM RecSys
Challenge 2018 for Automatic Music Playlist Continuation'

1. Using Matrix factorization methods to identify low dimensional latent representations of playlists and songs. Techniques used are Weighted Regularized Matrix Factorization (WRMF), LightFM, Weighted approximate rank pairwise (WARP) loss and Bayesian personalized ranking (BPR).
2. Collaborative Autoencoder / CNN ensemble. Song list, artist list, playlist title.
3. neighborhood-based
   collaborative filtering methods.
4. word2vec

Order, context and popularity bias in next-song recommendations
https://link.springer.com/article/10.1007/s13735-019-00169-8

1. Latent Markov Embedding
2. RNN

Are the conclusions from existing work similar to or different from your own findings?

--update with our findings

5.
