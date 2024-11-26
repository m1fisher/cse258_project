import csv
from dataclasses import dataclass

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
        tracks = [Track(x[0], x[1], x[2], x[3], x[4]) for x in reader]
    return tracks

