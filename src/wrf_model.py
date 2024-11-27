from functools import lru_cache
import threadpoolctl
import pickle

import implicit
import numpy as np
from sparse_repr import _load_sparse_mat
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from voyager import Index, Space

import utils


def run_svd():
    matrix = _load_sparse_mat("train_data")
    # Take the *transpose* of the matrix so that it is
    # |num songs| x |num playlists|. We are going to
    # project songs down into a low-dim subspace.
    matrix = matrix.T
    n_dimensions = 10
    svd = TruncatedSVD(n_components=n_dimensions)
    model = svd.fit(matrix)
    with open("train_data/svd.pkl", "wb") as fh:
        pickle.dump(model, fh)

def create_nearest_neighbor_idx():
    X = _load_X()
    index = Index(Space.Euclidean, num_dimensions=X.shape[1])
    i = 0
    for row in X:
        index.add_item(row)
        i += 1
        if (i % 10000) == 0:
            print(f"{i=}")
    index.save(os.path.join("train_data", "svd_voyager_index.voy"))

@lru_cache
def _load_voyager_index():
    return Index.load("train_data/svd_voyager_index.voy")

@lru_cache
def _load_model():
    with open("train_data/svd.pkl", "rb") as fh:
        model = pickle.load(fh)
    return model

@lru_cache
def _load_X():
    with open("train_data/svd.pkl", "rb") as fh:
        model = pickle.load(fh)
    X = model.transform(_load_sparse_mat("train_data").T)
    return X

def predict(playlists):
    """
    """
    track_to_artist = utils.get_track_to_artist_map("train_data/track_to_artist_ids.csv")
    model = _load_model()
    X = _load_X()
    preds = {}
    for pid, seed_tracks in playlists.items():
        track_ids = [t.track_id for t in seed_tracks]
        # try getting the 500 nearest neighbors of the first song in playlist, as a test
        #track_id_to_name = utils.get_track_id_to_name_map("train_data/track_ids.csv")
        recommendations = []
        for tid in track_ids:
            song = X[tid]
            distances = np.linalg.norm(X - song, axis=1)
            recommendations.extend(np.argsort(distances)[:50])
            print(len(recommendations))
        preds[pid] = [utils.Track(pid=pid, pos=None, track_id=int(x),
                                  artist_id=track_to_artist[int(x)], album_id=None)
                      for x in recommendations]
        continue
        ### ORIGINAL ATTEMPT ###
        # First, project the new playlist into the low-dimensional song space
        # We take the average song repr of the playlist
        vector = coo_matrix(([1] * len(track_ids), ([0] * len(track_ids), track_ids)), shape=(1, X.shape[0]))
        projected_vector = (vector @ X) / len(track_ids)
        # TODO: make sure this normalization is correct
        proj = projected_vector / np.linalg.norm(projected_vector)
        # Now, get the nearest neighbors of the playlist repr
        distances = np.linalg.norm(X - proj, axis=1)
        recommendations = np.argsort(distances)[:500]
        preds[pid] = [utils.Track(pid=pid, pos=None, track_id=int(x),
                                  artist_id=track_to_artist[int(x)], album_id=None)
                      for x in recommendations]
    return preds


if __name__ == "__main__":
    run_svd()
    #create_nearest_neighbor_idx()
