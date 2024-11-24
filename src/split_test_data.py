import json
import os
import random
import sys

random.seed(414)


def _make_data_slice(playlist_set, fetcher, data_name, output_data_dir):
    curr_json = []
    for playlist in playlist_set:
        curr_json.append(fetcher(playlist))
    with open(os.path.join(output_data_dir, f"{data_name}.json"), "w+") as fh:
        json.dump(curr_json, fh, indent=4)
    with open(
        os.path.join(output_data_dir, f"ground_truth_{data_name}.json"), "w+"
    ) as fh:
        json.dump(playlist_set, fh, indent=4)


def make_generator(lst, chunk_size):
    # c/o gpt4
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def main(data_file):
    output_data_dir = os.path.dirname(data_file)
    with open(data_file) as fh:
        test_json = json.load(fh)

    # NOTE: this sorting (or a similar approach) is necessary to ensure
    # that e.g. 100 and 25 track test playlists actually have enough tracks,
    # but it may introduce statistical biases into the test set.
    # TODO (mfisher): Implement a generator approach that shuffles the order
    # and then returns playlists with length >= min length for each split
    playlists = [x for x in test_json["playlists"]]
    playlists = sorted(playlists, key=lambda x: len(x["tracks"]), reverse=True)
    playlists = make_generator(playlists, chunk_size=1000)

    title_and_100_tracks = next(playlists)
    assert all(len(x["tracks"]) > 100 for x in title_and_100_tracks)
    fetch_title_and_100_tracks = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][:100],
    }
    _make_data_slice(
        title_and_100_tracks,
        fetch_title_and_100_tracks,
        "title_and_first_100_tracks",
        output_data_dir,
    )

    title_and_100_random_tracks = next(playlists)
    assert all(len(x["tracks"]) > 100 for x in title_and_100_random_tracks)
    fetch_title_and_100_random_tracks = lambda playlist: {
        "name": playlist["name"],
        "tracks": random.sample(playlist["tracks"], 100),
    }
    _make_data_slice(
        title_and_100_random_tracks,
        fetch_title_and_100_random_tracks,
        "title_and_random_100_tracks",
        output_data_dir,
    )

    title_and_first_25 = next(playlists)
    fetch_title_and_first_twenty_five = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][:25],
    }
    assert all(len(x["tracks"]) > 25 for x in title_and_first_25)
    _make_data_slice(
        title_and_first_25,
        fetch_title_and_first_twenty_five,
        "title_and_first_25_tracks",
        output_data_dir,
    )

    title_and_25_random_tracks = next(playlists)
    assert all(len(x["tracks"]) > 25 for x in title_and_25_random_tracks)
    fetch_title_and_25_random_tracks = lambda playlist: {
        "name": playlist["name"],
        "tracks": random.sample(playlist["tracks"], 25),
    }
    _make_data_slice(
        title_and_25_random_tracks,
        fetch_title_and_25_random_tracks,
        "title_and_random_25_tracks",
        output_data_dir,
    )

    title_and_first_ten = next(playlists)
    assert all(len(x["tracks"]) > 10 for x in title_and_first_ten)
    fetch_title_and_first_ten = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][:10],
    }
    _make_data_slice(
        title_and_first_ten,
        fetch_title_and_first_ten,
        "title_and_first_10_tracks",
        output_data_dir,
    )

    first_ten = next(playlists)
    assert all(len(x["tracks"]) > 10 for x in title_and_first_ten)
    fetch_first_ten = lambda playlist: {
        "tracks": playlist["tracks"][:10],
    }
    _make_data_slice(first_ten, fetch_first_ten, "first_10_tracks", output_data_dir)

    title_and_first_five = next(playlists)
    assert all(len(x["tracks"]) > 5 for x in title_and_first_ten)
    fetch_title_and_first_five = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][:5],
    }
    _make_data_slice(
        title_and_first_five,
        fetch_title_and_first_five,
        "title_and_first_5_tracks",
        output_data_dir,
    )

    first_five = next(playlists)
    assert all(len(x["tracks"]) > 5 for x in first_five)
    fetch_first_five = lambda playlist: {
        "tracks": playlist["tracks"][:5],
    }
    _make_data_slice(first_five, fetch_first_five, "first_5_tracks", output_data_dir)

    title_and_first_track = next(playlists)
    assert all(len(x["tracks"]) > 1 for x in title_and_first_track)
    fetch_title_and_first_track = lambda playlist: {
        "name": playlist["name"],
        "tracks": playlist["tracks"][0],
    }
    _make_data_slice(
        title_and_first_track,
        fetch_title_and_first_track,
        "title_and_first_1_track",
        output_data_dir,
    )

    title_only = next(playlists)
    assert all(len(x["tracks"]) > 1 for x in title_only)
    fetch_title_only = lambda playlist: {"name": playlist["name"]}
    _make_data_slice(title_only, fetch_title_only, "title_only", output_data_dir)


if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)
