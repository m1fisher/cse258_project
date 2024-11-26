import csv
import json
import os
import sys

def main(data_dir, out_dir):
    all_slices = os.listdir(data_dir)
    track_ids = {}
    curr_track_id = 0
    artist_ids = {}
    curr_artist_id = 0
    album_ids = {}
    curr_album_id = 0
    playlist_names = {}
    batch_number = 0
    track_id_to_artist_id = {}
    os.makedirs(out_dir, exist_ok=True)
    for slice_file in all_slices:
        batch_number += 1
        if batch_number % 10 == 0:
            print(f"processing batch number {batch_number}")
        with open(os.path.join(data_dir, slice_file)) as fh:
            curr_json = json.load(fh)
        playlists = curr_json["playlists"]
        processed_playlists = []
        for playlist in playlists:
            curr_playlist = []
            playlist_name = playlist["name"]
            pid = playlist["pid"]
            playlist_names[pid] = playlist_name
            for track in playlist["tracks"]:
                track_name = track["track_name"]
                track_id = track_ids.get(track_name)
                if track_id is None:
                    track_ids[track_name] = curr_track_id
                    track_id = track_ids[track_name]
                    curr_track_id += 1
                artist_name = track["artist_name"]
                artist_id = artist_ids.get(artist_name)
                if artist_id is None:
                    artist_ids[artist_name] = curr_artist_id
                    artist_id = artist_ids[artist_name]
                    curr_artist_id += 1
                album_name = track["album_name"]
                album_id = album_ids.get(album_name)
                if album_id is None:
                    album_ids[album_name] = curr_album_id
                    album_id = album_ids[album_name]
                    curr_album_id += 1
                track_id_to_artist_id[track_id] = artist_id
                pos = track["pos"]
                curr_playlist.append((pid, pos, track_id, artist_id, album_id))
            processed_playlists.extend(curr_playlist)

        file_path = slice_file.replace("json", "csv")
        # Write the list of tuples to a CSV file
        with open(os.path.join(out_dir, file_path), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pid', 'pos', 'track_id', 'artist_id', 'album_id'])
            writer.writerows(processed_playlists)

    write_ids(track_ids, "track", out_dir)
    write_ids(album_ids, "album", out_dir)
    write_ids(artist_ids, "artist", out_dir)
    write_ids({v: k for k, v in playlist_names.items()}, "playlist", out_dir)
    track_to_artist_tuples =  [(k, v) for k, v in track_id_to_artist_id.items()]
    with open(os.path.join(out_dir, f"track_to_artist_ids.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'track_id', f'artist_id'])
        writer.writerows(track_to_artist_tuples)


def write_ids(container, name, out_dir):
    container_tuples =  [(k, v) for k, v in container.items()]
    with open(os.path.join(out_dir, f"{name}_ids.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'{name}_name', f'{name}_id'])
        writer.writerows(container_tuples)



if __name__ == "__main__":
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    main(data_dir, out_dir)
