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
    """if playlist_rows = False, we do song rows"""
    matrix = _load_sparse_mat("train_data")
    if playlist_rows == False:
        # Take the *transpose* of the matrix so that it is
        # |num songs| x |num playlists|. We are going to
        # project songs down into a low-dim subspace.
        matrix = matrix.T
    n_dimensions = 200  # 200 at scale
    threadpoolctl.threadpool_limits(1, "blas")
    svd = implicit.cpu.als.AlternatingLeastSquares(
        factors=n_dimensions, regularization=0.001, alpha=100.0, random_state=414
    )
    svd.fit(matrix)
    with open(get_fname(playlist_rows), "wb") as fh:
        pickle.dump(svd, fh)


def get_fname(playlist_rows=True):
    name = "playlist_rows" if playlist_rows else "song_rows"
    filename = f"train_data/svd_{name}.pkl"
    return filename


@lru_cache
def _load_model(playlist_rows=True):
    with open(get_fname(playlist_rows), "rb") as fh:
        model = pickle.load(fh)
    return model


class LatentFactors:
    def __init__(self):
        self.model_playlist = _load_model()
        self.num_candidates = 1000

    def get_playlist_similarity_candidates(self, playlists):
        """ """
        track_to_artist = utils.get_track_to_artist_map(
            "train_data/track_to_artist_ids.csv"
        )
        track_to_album = utils.get_track_to_artist_map(
            "train_data/track_to_album_ids.csv"
        )
        model = self.model_playlist
        preds = {}
        pred_scores = {}
        i = 0
        for pid, seed_tracks in playlists.items():
            i += 1
            if (i % 10) == 0:
                print(i)
            track_ids = [t.track_id for t in seed_tracks]
            # compute a 'compatibility score' for each
            # song compared to the seed tracks.
            # The 'compatibility score' is the average inner
            # product of each song with all the seed_track songs.
            # c/o gpt4 for this initial code
            item_factors = model.item_factors[track_ids]
            scores = item_factors @ model.item_factors.T
            score_means = scores.mean(axis=0)
            ranked_items = np.argpartition(score_means, -self.num_candidates)[-self.num_candidates:]
            idxs = np.argsort(score_means[ranked_items])[::-1]
            recommendations = ranked_items[idxs]
            pred_scores[pid] = score_means[idxs]
            preds[pid] = [
                utils.Track(
                    pid=pid,
                    pos=i,
                    track_id=int(x),
                    artist_id=track_to_artist[int(x)],
                    album_id=track_to_album[int(x)],
                )
                for i, x in enumerate(recommendations)
            ]
        return preds, pred_scores

    def predict(self, playlists):
        plist_recs, _ = self.get_playlist_similarity_candidates(playlists)
        return {pid: plist_recs[pid] for pid in playlists}

    def predict_with_scores(self, playlists):
        return self.get_playlist_similarity_candidates(playlists)


if __name__ == "__main__":
    print("running svd for playlist rows")
    run_svd()
