import json
import os
import random
import sys

# Set seed for reproducibility
random.seed(414)

def make_test_data(data_dir, rm):
    all_files = os.listdir(data_dir)
    test_data_name = 'mpd.slice.{x}00000-{x}00999.json'
    test_files = [test_data_name.format(x=i) for i in range(1, 10)]
    test_files.append('mpd.slice.0-999.json')
    assert set(test_files).issubset(all_files)
    test_json = {'info': [], 'playlists': []}
    for f in test_files:
        with open(os.path.join(data_dir, f)) as fh:
            curr_json = json.load(fh)
            test_json['info'].append(curr_json['info'])
            test_json['playlists'].extend(curr_json['playlists'])
    # shuffle playlists for good measure
    # note that this does not actually shuffle the order
    # of each playlist's tracks, which is important to preserve.
    random.shuffle(test_json['playlists'])
    os.makedirs('test_data', exist_ok=True)
    with open(os.path.join('test_data', 'mpd.test.json'), "w+") as fh:
        json.dump(test_json, fh, indent=4)
    if rm:
        for f in test_files:
            os.remove(f)

def make_validation_data(data_dir, rm):
    all_files = os.listdir(data_dir)
    validation_data_prefix = 'mpd.slice.9'
    test_files = [test_data_name.format(x=i) for i in range(1, 10)]
    test_files.append('mpd.slice.0-999.json')
    assert set(test_files).issubset(all_files)
    test_json = {'info': [], 'playlists': []}
    for f in test_files:
        with open(os.path.join(data_dir, f)) as fh:
            curr_json = json.load(fh)
            test_json['info'].append(curr_json['info'])
            test_json['playlists'].extend(curr_json['playlists'])
    # shuffle playlists for good measure
    # note that this does not actually shuffle the order
    # of each playlist's tracks, which is important to preserve.
    random.shuffle(test_json['playlists'])
    os.makedirs('test_data', exist_ok=True)
    with open(os.path.join('test_data', 'mpd.test.json'), "w+") as fh:
        json.dump(test_json, fh, indent=4)
    if rm:
        for f in test_files:
            os.remove(f)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    mode = sys.argv[2]
    rm = False
    if len(sys.argv) > 3 and sys.argv[3] == "--rm":
        rm = True
    if mode == "test":
        make_test_data(data_dir, rm)
    elif mode == "validation":
        make_validation_data(data_dir, rm)
    else:
        raise ValueError(f"Invalid value provided for mode: {mode}")

