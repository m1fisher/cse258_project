from collections import defaultdict
import csv
from itertools import chain
import os
import random
import sys

from utils import Track, write_track_csv

random.seed(414)


def _make_data_slice(ground_truth, test_set, data_name, output_data_dir):
    write_track_csv(test_set, os.path.join(output_data_dir, f"{data_name}.csv"))
    write_track_csv(ground_truth, os.path.join(output_data_dir, f"ground_truth_{data_name}.csv"))


def make_generator(lst, chunk_size):
    # c/o gpt4
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def flatten(nested_list):
    return list(chain.from_iterable(nested_list))


def main(data_file):
    output_data_dir = os.path.dirname(data_file)
    with open(data_file) as fh:
       reader = csv.reader(fh)
       data = []
       for row in reader:
           if 'pid' not in row:  # disregard header
               data.append(Track(*(int(x) for x in row)))
    playlist_map = defaultdict(list)
    for track in data:
        playlist_map[track.pid].append(track)
    playlists = [p for p in playlist_map.values()]

    # NOTE: this sorting (or a similar approach) is necessary to ensure
    # that e.g. 100 and 25 track test playlists actually have enough tracks.
    playlists = sorted(playlists, key=len, reverse=True)
    playlists = make_generator(playlists, chunk_size=1000)

    title_and_first_100_tracks = next(playlists)
    assert all(len(x) > 100 for x in title_and_first_100_tracks)
    test = flatten([x[:100] for x in title_and_first_100_tracks])
    ground_truth = flatten([x[100:] for x in title_and_first_100_tracks])
    _make_data_slice(
        ground_truth,
        test,
        "title_and_first_100_tracks",
        output_data_dir,
    )

    title_and_100_random_tracks = next(playlists)
    for p in title_and_100_random_tracks:
        random.shuffle(p)
    assert all(len(x) > 100 for x in title_and_100_random_tracks)
    test = flatten([x[:100] for x in title_and_100_random_tracks])
    ground_truth = flatten([x[100:] for x in title_and_100_random_tracks])
    _make_data_slice(
        ground_truth,
        test,
        "title_and_random_100_tracks",
        output_data_dir,
    )

    title_and_first_25_tracks = next(playlists)
    assert all(len(x) > 25 for x in title_and_first_25_tracks)
    test = flatten([x[:25] for x in title_and_first_25_tracks])
    ground_truth = flatten([x[25:] for x in title_and_first_25_tracks])
    _make_data_slice(
        ground_truth,
        test,
        "title_and_first_25_tracks",
        output_data_dir,
    )

    title_and_25_random_tracks = next(playlists)
    for p in title_and_25_random_tracks:
        random.shuffle(p)
    assert all(len(x) > 25 for x in title_and_25_random_tracks)
    test = flatten([x[:25] for x in title_and_25_random_tracks])
    ground_truth = flatten([x[25:] for x in title_and_25_random_tracks])
    _make_data_slice(
        ground_truth,
        test,
        "title_and_random_25_tracks",
        output_data_dir,
    )

    title_and_first_10_tracks = next(playlists)
    assert all(len(x) > 10 for x in title_and_first_10_tracks)
    test = flatten([x[:10] for x in title_and_first_10_tracks])
    ground_truth = flatten([x[10:] for x in title_and_first_10_tracks])
    _make_data_slice(
        ground_truth,
        test,
        "title_and_first_10_tracks",
        output_data_dir,
    )

    fake_pids = set()

    first_ten = next(playlists)
    assert all(len(x) > 10 for x in first_ten)
    for p in first_ten:
        # Scrub the playlist ID
        fake_pid = random.randint(int(-1e7), -1)
        while fake_pid in fake_pids:
            fake_pid = random.randint(int(-1e7), -1)
        for x in p:
            x.pid = fake_pid
            assert x.pid not in fake_pids
        fake_pids.add(x.pid)
    test = flatten([x[:10] for x in first_ten])
    ground_truth = flatten([x[10:] for x in first_ten])
    _make_data_slice(
        ground_truth,
        test,
        "first_10_tracks",
        output_data_dir,
    )

    title_and_first_5_tracks = next(playlists)
    assert all(len(x) > 5 for x in title_and_first_5_tracks)
    test = flatten([x[:5] for x in title_and_first_5_tracks])
    ground_truth = flatten([x[5:] for x in title_and_first_5_tracks])
    _make_data_slice(
        ground_truth,
        test,
        "title_and_first_5_tracks",
        output_data_dir,
    )

    first_5 = next(playlists)
    assert all(len(x) > 5 for x in first_5)
    for p in first_5:
        # Scrub the playlist ID
        fake_pid = random.randint(int(-1e7), -1)
        while fake_pid in fake_pids:
            fake_pid = random.randint(int(-1e7), -1)
        for x in p:
            x.pid = fake_pid
            assert x.pid not in fake_pids
        fake_pids.add(x.pid)
    test = flatten([x[:5] for x in first_5])
    ground_truth = flatten([x[5:] for x in first_5])
    _make_data_slice(
        ground_truth,
        test,
        "first_5_tracks",
        output_data_dir,
    )


    title_and_first_1_tracks = next(playlists)
    assert all(len(x) > 1 for x in title_and_first_1_tracks)
    test = flatten([x[:1] for x in title_and_first_1_tracks])
    ground_truth = flatten([x[1:] for x in title_and_first_1_tracks])
    _make_data_slice(
        ground_truth,
        test,
        "title_and_first_1_tracks",
        output_data_dir,
    )

    title_only = next(playlists)
    assert all(len(x) > 1 for x in title_only)
    title_only = flatten(title_only)
    pid_set = set(x.pid for x in title_only)
    pids = [Track(pid, None, None, None, None) for pid in pid_set]
    _make_data_slice(pids, title_only, "title_only", output_data_dir)


if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)
