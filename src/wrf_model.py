from functools import lru_cache
import threadpoolctl
import pickle

import implicit
from sparse_repr import _load_sparse_mat
from scipy.sparse import coo_matrix

import utils

def main():
    matrix = _load_sparse_mat("train_data")
    matrix = matrix.tocsr()
    threadpoolctl.threadpool_limits(1, "blas")
    model = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.1, iterations=20)
    model.fit(matrix)
    with open("train_data/wrmf.pkl", "wb") as fh:
        pickle.dump(model, fh)

@lru_cache
def _load_model():
    with open("train_data/wrmf.pkl", "rb") as fh:
        model = pickle.load(fh)
    return model

def predict(playlists):
    """
    """
    track_to_artist = utils.get_track_to_artist_map("train_data/track_to_artist_ids.csv")
    model = _load_model()
    preds = {}
    for pid, seed_tracks in playlists.items():
        track_ids = [t.track_id for t in seed_tracks]
        vector = coo_matrix(([1] * len(track_ids), ([0] * len(track_ids), track_ids)), shape=(1, len(model.item_factors)))
        vector = vector.tocsr()
        import pdb; pdb.set_trace()

        recommendations = model.recommend(-1, vector, N=500)[0]
        preds[pid] = [utils.Track(pid=pid, pos=None, track_id=int(x),
                                  artist_id=track_to_artist[int(x)], album_id=None)
                      for x in recommendations]
    return preds


if __name__ == "__main__":
    main()
