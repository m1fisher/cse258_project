# CSE258 Assignment 2

Original challenge description [here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

Actually downloaded dataset [here](https://www.kaggle.com/datasets/himanshuwagh/spotify-million)

Useful references:

    * An Analysis of Approaches Taken in the ACM RecSys Challenge 2018 for Automatic Music Playlist Continuation (in repo)
    * http://www.cs.utoronto.ca/~mvolkovs/recsys2018_challenge.pdf
    * Try this model: https://spotify.github.io/voyager/python/reference.html 
    * Try this model: https://www.cs.cornell.edu/~tj/publications/chen_etal_12a.pdf

## Cookbook

Follow steps below or execute cookbook.sh
```
chmod +x cookbook.sh
./cookbook.sh
```

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
# Create the sparse matrix representation
python3 src/sparse_repr.py train_data
# Run latent factor SVD matrix factorization
python3 src/latent_factor_model.py
# Run the model blend to create ranker training
python3 src/model_blend.py train_data
```

