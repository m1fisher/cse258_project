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


def run_svd(playlist_rows=True):
    """ if playlist_rows = False, we do song rows """
    matrix = _load_sparse_mat("train_data")
    if playlist_rows == False:
        # Take the *transpose* of the matrix so that it is
        # |num songs| x |num playlists|. We are going to
        # project songs down into a low-dim subspace.
        matrix = matrix.T
    n_dimensions = 100
    svd = TruncatedSVD(n_components=n_dimensions, random_state=414)
    model = svd.fit(matrix)
    with open(get_fname(playlist_rows), "wb") as fh:
        pickle.dump(model, fh)

def get_fname(playlist_rows=True):
    name = "playlist_rows" if playlist_rows else "song_rows"
    filename = f"train_data/svd_{name}.pkl"
    return filename

#def create_nearest_neighbor_idx():
#    X = _load_X()
#    index = Index(Space.Euclidean, num_dimensions=X.shape[1])
#    i = 0
#    for row in X:
#        index.add_item(row)
#        i += 1
#        if (i % 10000) == 0:
#            print(f"{i=}")
#    index.save(os.path.join("train_data", "svd_voyager_index.voy"))
#
#@lru_cache
#def _load_voyager_index():
#    return Index.load("train_data/svd_voyager_index.voy")

@lru_cache
def _load_model(playlist_rows=True):
    with open(get_fname(playlist_rows), "rb") as fh:
        model = pickle.load(fh)
    return model

@lru_cache
def _load_X(playlist_rows=True):
    with open(get_fname(playlist_rows), "rb") as fh:
        model = pickle.load(fh)
    sparse_mat = _load_sparse_mat("train_data")
    if playlist_rows == False:
        sparse_mat = sparse_mat.T
    X = model.transform(sparse_mat)
    return X

class LatentFactors:
    def __init__(self):
        self.model_playlist = _load_model()
        self.X_playlist = _load_X()
        self.model_songs = _load_model(playlist_rows=False)
        self.X_songs = _load_X(playlist_rows=False)

    def get_song_similarity_candidates(self, playlists):
        """
        """
        track_to_artist = utils.get_track_to_artist_map("train_data/track_to_artist_ids.csv")
        model = self.model_songs
        X = self.X_songs
        X_T = X.T
        preds = {}
        i = 0
        for pid, seed_tracks in playlists.items():
            i += 1
            track_ids = [t.track_id for t in seed_tracks]
            # First, project the new playlist into the low-dimensional song space
            # We take the average song repr of the playlist
            proj = sum(X[i] for i in track_ids) / len(track_ids)
            # Now, get the nearest neighbors of the playlist repr
            distances = np.linalg.norm(X - proj, axis=1)
            # TODO: remove seed tracks from recommendations
            recommendations = np.argsort(distances)[:5000]
            preds[pid] = [utils.Track(pid=pid, pos=None, track_id=int(x),
                                      artist_id=track_to_artist[int(x)], album_id=None)
                          for x in recommendations]
        return preds

    # TODO: dedup this with song similarity function
    def get_playlist_similarity_candidates(self, playlists):
        """
        """
        track_to_artist = utils.get_track_to_artist_map("train_data/track_to_artist_ids.csv")
        model = self.model_playlist
        X = self.X_playlist
        X_T = X.T
        preds = {}
        i = 0
        for pid, seed_tracks in playlists.items():
            print(i)
            i += 1
            # TODO: get this dimension
            track_id_set = set(t.track_id for t in seed_tracks)
            playlist_vector = np.array([1 if i in track_id_set else 0 for i in
                                        range(model.components_.shape[1])])
            # First, project the new playlist into the low-dimensional song space
            # c/0 gpt4 for this line of projection code
            proj = np.dot(playlist_vector, model.components_.T)
            # Now, get the nearest neighbors of the playlist repr
            distances = np.linalg.norm(X - proj, axis=1)
            recommendations = np.argsort(distances)[:5000]
            preds[pid] = [utils.Track(pid=pid, pos=None, track_id=int(x),
                                      artist_id=track_to_artist[int(x)], album_id=None)
                          for x in recommendations]
        return preds

    def predict(self, playlists):
        # TODO process this right
        plist_recs = self.get_playlist_similarity_candidates(playlists)
        song_recs = self.get_song_similarity_candidates(playlists)
        return {pid: plist_recs[pid] + song_recs[pid] for pid in playlists}

if __name__ == "__main__":
    print("running svd for playlist rows")
    run_svd()
    print("running svd for song rows")
    run_svd(playlist_rows=False)
    #create_nearest_neighbor_idx()
