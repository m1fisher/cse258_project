import csv
import os
import random
import shutil
import sys

# Set seed for reproducibility
random.seed(414)

VALIDATION_FILES = [
    "mpd.slice.954000-954999.csv",
    "mpd.slice.955000-955999.csv",
    "mpd.slice.956000-956999.csv",
    "mpd.slice.957000-957999.csv",
    "mpd.slice.958000-958999.csv",
    "mpd.slice.959000-959999.csv",
    "mpd.slice.96000-96999.csv",
    "mpd.slice.960000-960999.csv",
    "mpd.slice.961000-961999.csv",
    "mpd.slice.962000-962999.csv",
    "mpd.slice.963000-963999.csv",
    "mpd.slice.964000-964999.csv",
    "mpd.slice.965000-965999.csv",
    "mpd.slice.966000-966999.csv",
    "mpd.slice.967000-967999.csv",
    "mpd.slice.968000-968999.csv",
    "mpd.slice.969000-969999.csv",
    "mpd.slice.97000-97999.csv",
    "mpd.slice.970000-970999.csv",
    "mpd.slice.971000-971999.csv",
    "mpd.slice.972000-972999.csv",
    "mpd.slice.973000-973999.csv",
    "mpd.slice.974000-974999.csv",
    "mpd.slice.975000-975999.csv",
    "mpd.slice.976000-976999.csv",
    "mpd.slice.977000-977999.csv",
    "mpd.slice.978000-978999.csv",
    "mpd.slice.979000-979999.csv",
    "mpd.slice.98000-98999.csv",
    "mpd.slice.980000-980999.csv",
    "mpd.slice.981000-981999.csv",
    "mpd.slice.982000-982999.csv",
    "mpd.slice.983000-983999.csv",
    "mpd.slice.984000-984999.csv",
    "mpd.slice.985000-985999.csv",
    "mpd.slice.986000-986999.csv",
    "mpd.slice.987000-987999.csv",
    "mpd.slice.988000-988999.csv",
    "mpd.slice.989000-989999.csv",
    "mpd.slice.99000-99999.csv",
    "mpd.slice.990000-990999.csv",
    "mpd.slice.991000-991999.csv",
    "mpd.slice.992000-992999.csv",
    "mpd.slice.993000-993999.csv",
    "mpd.slice.994000-994999.csv",
    "mpd.slice.995000-995999.csv",
    "mpd.slice.996000-996999.csv",
    "mpd.slice.997000-997999.csv",
    "mpd.slice.998000-998999.csv",
    "mpd.slice.999000-999999.csv",
]


def make_test_data(data_dir, mv):
    all_files = os.listdir(data_dir)
    test_data_name = "mpd.slice.{x}00000-{x}00999.csv"
    test_files = [test_data_name.format(x=i) for i in range(1, 10)]
    test_files.append("mpd.slice.0-999.csv")
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
