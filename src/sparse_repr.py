from functools import lru_cache
import os
import pickle
import sys

import numpy as np
from scipy.sparse import coo_matrix

import utils

def create_sparse_mat(data_dir):
    slices = [x for x in os.listdir(data_dir) if x.startswith("mpd.slice")]
    mat_rows = {}
    batch_count = 0
    for filename in slices:
        batch_count += 1
        if (batch_count % 10) == 0:
            print(f"processing batch {batch_count}")
        tracks = utils.read_track_csv(os.path.join(data_dir, filename))
        for track in tracks:
            if track.pid not in mat_rows:
                mat_rows[track.pid] = [track.track_id]
            else:
                mat_rows[track.pid].append(track.track_id)
    rows = []
    cols = []
    for row_idx, row in enumerate(mat_rows.values()):
        for col in row:
            rows.append(row_idx)
            cols.append(col)
    # Binary matrix
    values = [1] * len(cols)
    matrix = coo_matrix((values, (rows, cols)))
    pickle.dump(matrix, open(os.path.join(data_dir, "sparse_matrix.pkl"), "wb"))
    print("Complete")

@lru_cache
def _load_sparse_mat(data_dir):
    with open(os.path.join(data_dir, "sparse_matrix.pkl"), "rb") as fh:
        sparse_mat = pickle.load(fh)
    return sparse_mat

@lru_cache
def _load_normalized_sparse_mat(data_dir):
    matrix = _load_sparse_mat(data_dir)
    # divide rows by their magnitudes, c/o gpt4
    squared_data = matrix.data ** 2
    row_sums = np.bincount(matrix.row, weights=squared_data, minlength=matrix.shape[0])
    row_magnitudes = np.sqrt(row_sums)
    row_magnitudes[row_magnitudes == 0] = 1  # Prevent division by zero
    normalized_data = matrix.data / row_magnitudes[matrix.row]
    normalized_matrix = coo_matrix((normalized_data, (matrix.row, matrix.col)), shape=matrix.shape)
    # TODO (mfisher): Inverse scale entries by popularity
    return normalized_matrix


class NeighborModels:
    def __init__(self):
        self.ns_mat = _load_normalized_sparse_mat("train_data")
        self.ns_mat_csr = self.ns_mat.tocsr()
        self.ns_mat_csc = self.ns_mat.tocsc()

    def user_to_user_score(self, i, j: int):
        """
        i is a playlist, j is a song.
        idea c/o http://www.cs.utoronto.ca/~mvolkovs/recsys2018_challenge.pdf
        """
        assert len(set(i.row)) == 1
        # normalize i vector
        i = i / np.sqrt((i.data ** 2).sum())
        # Find playlists that include song j (rows with column j non-zero)
        start, end = self.ns_mat_csc.indptr[j], self.ns_mat_csc.indptr[j + 1]
        U_j = self.ns_mat_csc.indices[start:end]  # Rows where column j is non-zero
        # Exclude i itself if it's part of U_j
        U_j = U_j[U_j != i.row[0]]
        # c/o gpt4 for optimized code
        # Compute the total inner product in a vectorized way
        # Slice all relevant rows at once and perform the dot product
        relevant_rows = self.ns_mat_csr[U_j]
        total_inner_product = relevant_rows.dot(i.T).sum()

        return total_inner_product

    def _get_col(self, mat, j: int):
        mask = (mat.col == j)
        rows, cols, data = mat.row[mask], mat.col[mask], mat.data[mask]
        return coo_matrix((data, (rows, [0]*len(data))), shape=(mat.shape[0], 1))

    def item_to_item_score(self, i, j: int):
        """
        i is a playlist, j is a song.
        idea c/o http://www.cs.utoronto.ca/~mvolkovs/recsys2018_challenge.pdf
        """
        # normalize i vector
        i = i / np.sqrt((i.data ** 2).sum())
        # Find all songs included in playlist i
        V_j = i.col  # Songs in playlist i (non-zero columns of i)
        # Exclude song j itself from comparisons
        V_j = V_j[V_j != j]
        # c/o gpt4 for optimized csc code
        # Get the column vector for song j
        j_col = self.ns_mat_csc[:, j]
        # Get all relevant columns (songs in V_j)
        relevant_cols = self.ns_mat_csc[:, V_j]
        # Compute dot products in bulk
        total_inner_product = relevant_cols.T.dot(j_col).sum()
        return total_inner_product


def inner_product_predict(playlists):
    """
    Playlist-to-playlist similarity baseline predictor.
    """
    sparse_mat = _load_normalized_sparse_mat("train_data")
    track_to_artist = utils.get_track_to_artist_map("train_data/track_to_artist_ids.csv")
    preds = {}
    for pid, seed_tracks in playlists.items():
        track_ids = [t.track_id for t in seed_tracks]
        val = 1 / np.sqrt((len(track_ids)))
        vector = coo_matrix(([val] * len(track_ids), ([0] * len(track_ids), track_ids)), shape=(1, sparse_mat.shape[1]))
        product = sparse_mat.dot(vector.transpose()).tocoo()
        row_tuples = [(r, v) for r,v in zip(product.row, product.data)]
        row_tuples.sort(key=lambda x: x[1], reverse=True)
        pred_set = set(track_ids)
        curr_preds = []
        i = 0
        while len(curr_preds) < 500:
            curr_playlist = row_tuples[i][0]
            track_ids = sparse_mat.col[sparse_mat.row == curr_playlist]
            for tid in track_ids:
                # NOTE: This is not ordered by item similarity
                if tid not in pred_set:
                    curr_preds.append(tid)
                    pred_set.add(tid)
            i += 1
        print(len(curr_preds))
        preds[pid] = [utils.Track(pid=pid, pos=None, track_id=int(x),
                                  artist_id=track_to_artist[int(x)], album_id=None)
                      for x in curr_preds[:500]]
    return preds


if __name__ == "__main__":
    data_dir = sys.argv[1]
    #create_sparse_mat(data_dir)
    nm = NeighborModels()
    test_plist = nm.ns_mat_csr[0].tocoo()
    for j in range(10000):
        nm.item_to_item_score(test_plist, j)
        nm.user_to_user_score(test_plist, j)
