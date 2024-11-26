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
