from collections import defaultdict
from functools import lru_cache
import csv
import json
import os
import sys

from utils import Track, read_track_csv

# TODO (mfisher): Include artists in popularity list

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


def write_popular_ids():
    with open("eda/top500.csv") as fh:
        reader = csv.reader(fh)
        next(reader)
        track_names = [x[1] for x in reader]
    name_to_id = {}
    with open("train_data/track_ids.csv") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            name_to_id[row[0]] = row[1]
    id_list = [int(name_to_id[x]) for x in track_names]
    with open('src/top500_ids.txt', 'w') as file:
        file.write('\n'.join(map(str, id_list)))

@lru_cache
def read_popular_ids():
    with open('src/top500_ids.txt', 'r') as file:
        popular_ids = [int(line.strip()) for line in file]
    return popular_ids

def model(playlists):
    popular_ids = read_popular_ids()
    tracks = [Track(pid=None, pos=None, track_id=id_, artist_id=None, album_id=None) for id_ in popular_ids]
    return {pid: tracks for pid in playlists}





if __name__ == "__main__":
    #train_data_dir = sys.argv[1]
    #calc_popularities(train_data_dir)
    write_popular_ids()
