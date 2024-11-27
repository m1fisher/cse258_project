from functools import lru_cache
import os
import sys


from voyager import Index, Space

import utils

NUM_TRACKS = 1483760
def create_voyager_index(data_dir):
    slices = [x for x in os.listdir(data_dir) if x.startswith("mpd.slice")]
    index = Index(Space.Euclidean, num_dimensions=NUM_TRACKS)
    batch_count = 0
    for filename in slices:
        batch_count += 1
        if (batch_count % 10) == 0:
            print(f"processing batch {batch_count}")
        tracks = utils.read_track_csv(os.path.join(data_dir, filename))
        pid_to_tracks = {}
        for t in tracks:
            if t.pid not in pid_to_tracks:
                pid_to_tracks[t.pid] = [t.track_id]
            else:
                pid_to_tracks[t.pid].append(t.track_id)
        vectors = []
        for playlist in pid_to_tracks.values():
            print("new playlist")
            track_ids = set(playlist)
            # NOTE: This is too slow too run over all playlists. Need to reduce dimensionality
            # or find a nearest-neighbor algorithm that supports sparse representation.
            index.add_item([1 if i in track_ids else 0 for i in range(NUM_TRACKS)])
        import pdb; pdb.set_trace()

    index.save(os.path.join(data_dir, "voyager_index.voy"))

@lru_cache
def _load_voyager_index():
    return Index.load("train_data/voyager_index.voy")

def predict(playlists):
    """
    Nearest-neighbor playlist predictor
    """
    voyager_index = _load_voyager_index()
    track_to_artist = utils.get_track_to_artist_map("train_data/track_to_artist_ids.csv")
    preds = {}
    for pid, seed_tracks in playlists.items():
        track_ids = set(x.track_id for x in seed_tracks)
        vector = [1 if i in track_ids else 0 for i in range(NUM_TRACKS)]
        nearest_neighbors, distances = voyager_index.query(vector, k=50)
        import pdb; pdb.set_trace()
        pred_set = set(track_ids)
        curr_preds = []
        i = 0
        while len(curr_preds) < 500:
            curr_playlist = voyager_index[nearest_neighbors[i]]
            track_ids = [i for i in range(len(curr_playlist)) if curr_playlist[i] != 0]
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
    create_voyager_index(data_dir)
