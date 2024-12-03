import csv
import os
import random
import shutil
import sys

# Set seed for reproducibility
random.seed(414)

# Randomly chosen slices
VALIDATION_FILES = ['mpd.slice.829000-829999.csv', 'mpd.slice.415000-415999.csv', 'mpd.slice.502000-502999.csv', 'mpd.slice.742000-742999.csv', 'mpd.slice.56000-56999.csv', 'mpd.slice.959000-959999.csv', 'mpd.slice.826000-826999.csv', 'mpd.slice.389000-389999.csv', 'mpd.slice.295000-295999.csv', 'mpd.slice.393000-393999.csv', 'mpd.slice.125000-125999.csv', 'mpd.slice.19000-19999.csv', 'mpd.slice.997000-997999.csv', 'mpd.slice.437000-437999.csv', 'mpd.slice.336000-336999.csv', 'mpd.slice.345000-345999.csv', 'mpd.slice.515000-515999.csv', 'mpd.slice.101000-101999.csv', 'mpd.slice.260000-260999.csv', 'mpd.slice.988000-988999.csv', 'mpd.slice.441000-441999.csv', 'mpd.slice.558000-558999.csv', 'mpd.slice.766000-766999.csv', 'mpd.slice.818000-818999.csv', 'mpd.slice.279000-279999.csv', 'mpd.slice.700000-700999.csv', 'mpd.slice.506000-506999.csv', 'mpd.slice.25000-25999.csv', 'mpd.slice.806000-806999.csv', 'mpd.slice.357000-357999.csv']
TEST_FILES = ['mpd.slice.723000-723999.csv', 'mpd.slice.337000-337999.csv', 'mpd.slice.749000-749999.csv', 'mpd.slice.844000-844999.csv', 'mpd.slice.418000-418999.csv', 'mpd.slice.434000-434999.csv', 'mpd.slice.762000-762999.csv', 'mpd.slice.763000-763999.csv', 'mpd.slice.605000-605999.csv', 'mpd.slice.222000-222999.csv']

def make_test_data(data_dir, mv):
    all_files = os.listdir(data_dir)
    test_files = TEST_FILES
    assert set(test_files).issubset(all_files)
    test_csv = []
    for f in test_files:
        with open(os.path.join(data_dir, f)) as fh:
            reader = csv.reader(fh)
            curr_csv = []
            for row in reader:
                # Convert each row to a tuple and append to the list
                if 'pid' not in row:  # disregard header
                    curr_csv.append(tuple(row))
            test_csv.append(curr_csv)
    out_dir = "test_data"
    filename = "mpd.test.csv"
    make_output_file(test_csv, test_files, out_dir, filename)


def make_validation_data(data_dir, mv):
    all_files = sorted(os.listdir(data_dir))
    # Use the 50 files (50k playlists) for validation
    validation_files = VALIDATION_FILES
    validation_csv = []
    for f in validation_files:
        with open(os.path.join(data_dir, f)) as fh:
            reader = csv.reader(fh)
            curr_csv = []
            for row in reader:
                # Convert each row to a tuple and append to the list
                if 'pid' not in row:  # disregard header
                    curr_csv.append(tuple(row))
            validation_csv.append(curr_csv)
    out_dir = "validation_data"
    filename = "mpd.validation.csv"
    make_output_file(validation_csv, validation_files, out_dir, filename)


def make_output_file(csv_list, file_list, out_dir, filename):
    # shuffle playlists for good measure
    # note that this does not actually shuffle the order
    # of each playlist's tracks, which is important to preserve.
    random.shuffle(csv_list)
    os.makedirs(out_dir, exist_ok=True)
    all_rows = [('pid', 'pos', 'track_id', 'artist_id', 'album_id')]
    for one_csv in csv_list:
        all_rows.extend(one_csv)
    with open(os.path.join(out_dir, filename), "w+") as fh:
        writer = csv.writer(fh)
        writer.writerows(all_rows)
    if mv:
        for f in file_list:
            shutil.move(os.path.join(data_dir, f), os.path.join(out_dir, f))


if __name__ == "__main__":
    data_dir = sys.argv[1]
    mode = sys.argv[2]
    mv = False
    if len(sys.argv) > 3 and sys.argv[3] == "--mv":
        mv = True
    if mode == "test":
        make_test_data(data_dir, mv)
    elif mode == "validation":
        make_validation_data(data_dir, mv)
    else:
        raise ValueError(f"Invalid value provided for mode: {mode}")
