import json
import os
import random
import sys

# Set seed for reproducibility
random.seed(414)

def main(data_dir, rm):
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
    with open(os.path.join(data_dir, 'mpd.test.json'), "w+") as fh:
        json.dump(test_json, fh, indent=4)
    if rm:
        for f in test_files:
            os.remove(f)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    rm = False
    if len(sys.argv) > 2 and sys.argv[2] == "--rm":
        rm = True
    main(data_dir, rm)

