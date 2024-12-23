import csv
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class Track:
    pid: int
    pos: int
    track_id: int
    artist_id: int
    album_id: int

def write_track_csv(tracks: list[Track], file_path: str):
    with open(file_path, 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pid', 'pos', 'track_id', 'artist_id', 'album_id'])
        writer.writerows([
            (x.pid, x.pos, x.track_id, x.artist_id, x.album_id)
            for x in tracks
        ])

def read_track_csv(file_path: str):
    with open(file_path) as file:
        reader = csv.reader(file)
        next(reader)  # disregard header row
        tracks = [Track(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4])) for x in reader]
    return tracks

@lru_cache
def get_track_to_artist_map(file_path: str):
    with open(file_path) as file:
        reader = csv.reader(file)
        next(reader)
        track_to_artist = {int(row[0]): int(row[1]) for row in reader}
    return track_to_artist

@lru_cache
def get_track_id_to_name_map(file_path: str):
    with open(file_path) as file:
        reader = csv.reader(file)
        next(reader)
        track_id_to_name = {int(row[1]): row[0] for row in reader}
    return track_id_to_name
