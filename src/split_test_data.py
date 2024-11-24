import json
import os
import random
import sys

random.seed(414)


def _make_data_slice(playlist_set, fetcher, data_name, test_data_dir):
    curr_json = []
    for playlist in playlist_set:
        curr_json.append(fetcher(playlist))
    with open(os.path.join(test_data_dir, f"{data_name}.json"), "w+") as fh:
        json.dump(curr_json, fh, indent=4)
    with open(
        os.path.join(test_data_dir, f"ground_truth_{data_name}.json"), "w+"
    ) as fh:
        json.dump(playlist_set, fh, indent=4)


def main(data_file):
    test_data_dir = os.path.dirname(data_file)
    with open(data_file) as fh:
        test_json = json.load(fh)
    hundred_track_playlists = [
        x for x in test_json["playlists"] if len(x["tracks"]) > 100
    ]
    random.shuffle(hundred_track_playlists)
    short_playlists = [x for x in test_json["playlists"] if len(x["tracks"]) <= 100]
    random.shuffle(short_playlists)

    title_and_100_tracks = hundred_track_playlists[:1000]
    fetch_title_and_100_tracks = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][:100],
    }
    _make_data_slice(
        title_and_100_tracks,
        fetch_title_and_100_tracks,
        "title_and_first_100_tracks",
        test_data_dir,
    )

    title_and_100_random_tracks = hundred_track_playlists[1000:2000]
    fetch_title_and_100_random_tracks = lambda playlist: {
        "name": playlist["name"],
        "tracks": random.sample(playlist["tracks"], 100),
    }
    _make_data_slice(
        title_and_100_random_tracks,
        fetch_title_and_100_random_tracks,
        "title_and_random_100_tracks",
        test_data_dir,
    )

    playlists = short_playlists + hundred_track_playlists[2000:]
    random.shuffle(playlists)

    title_only = lambda playlist: {"name": playlist["name"]}
    _make_data_slice(playlists[:1000], title_only, "title_only", test_data_dir)

    title_and_first_five = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][:5],
    }
    _make_data_slice(playlists[1000:2000], title_and_first_five, "title_and_first_5_tracks", test_data_dir)

    first_five = lambda playlist: {
        "tracks": playlist["tracks"][:5],
    }
    _make_data_slice(playlists[2000:3000], first_five, "first_5_tracks", test_data_dir)

    title_and_first_ten = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][:10],
    }
    _make_data_slice(playlists[3000:4000], title_and_first_ten, "title_and_first_10_tracks", test_data_dir)

    first_ten = lambda playlist: {
        "tracks": playlist["tracks"][:10],
    }
    _make_data_slice(playlists[4000:5000], first_ten, "first_10_tracks", test_data_dir)

    title_and_first_twenty_five = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][:25],
    }
    _make_data_slice(playlists[5000:6000], title_and_first_twenty_five, "title_and_first_25_tracks", test_data_dir)

    title_and_25_random_tracks = hundred_track_playlists[6000:7000]
    fetch_title_and_25_random_tracks = lambda playlist: {
        "name": playlist["name"],
        "tracks": random.sample(playlist["tracks"], 25),
    }
    _make_data_slice(
        title_and_25_random_tracks,
        fetch_title_and_25_random_tracks,
        "title_and_random_25_tracks",
        test_data_dir,
    )

    title_and_first_track = hundred_track_playlists[7000:8000]
    fetch_title_and_first_track = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][0]
    }
    _make_data_slice(
        title_and_first_track,
        fetch_title_and_first_track,
        "title_and_first_1_track",
        test_data_dir,
    )

if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)
