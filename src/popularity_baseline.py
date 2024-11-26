from collections import defaultdict
import csv
import json
import os
import sys

def calc_popularities(train_data_dir):
    all_slices = os.listdir(train_data_dir)
    assert all(x.startswith("mpd.slice") for x in all_slices)
    track_popularities = defaultdict(int)
    slice_num = 0
    for filename in all_slices:
        with open(os.path.join(train_data_dir, filename)) as fh:
            curr_json = json.load(fh)
        for playlist in curr_json['playlists']:
            for track in playlist['tracks']:
                track_popularities[track['track_name']] += 1
        slice_num += 1
        if (slice_num % 10) == 0:
            print(f"processed slice {slice_num}")
    track_tuples = [(v, k) for k,v in track_popularities.items()]
    track_tuples = sorted(track_tuples, reverse=True)
    file_path = 'all_popularities.csv'
    # Write the list of tuples to a CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frequency', 'track_name'])
        writer.writerows(track_tuples)
    print(f"All popularities saved to {file_path}")


def model():
    """
    Predict the 500 most popular tracks
    """
    return




if __name__ == "__main__":
    train_data_dir = sys.argv[1]
    calc_popularities(train_data_dir)
