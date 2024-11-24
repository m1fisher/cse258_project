import json
import os
import random
import shutil
import sys

# Set seed for reproducibility
random.seed(414)

VALIDATION_FILES = [
    "mpd.slice.954000-954999.json",
    "mpd.slice.955000-955999.json",
    "mpd.slice.956000-956999.json",
    "mpd.slice.957000-957999.json",
    "mpd.slice.958000-958999.json",
    "mpd.slice.959000-959999.json",
    "mpd.slice.96000-96999.json",
    "mpd.slice.960000-960999.json",
    "mpd.slice.961000-961999.json",
    "mpd.slice.962000-962999.json",
    "mpd.slice.963000-963999.json",
    "mpd.slice.964000-964999.json",
    "mpd.slice.965000-965999.json",
    "mpd.slice.966000-966999.json",
    "mpd.slice.967000-967999.json",
    "mpd.slice.968000-968999.json",
    "mpd.slice.969000-969999.json",
    "mpd.slice.97000-97999.json",
    "mpd.slice.970000-970999.json",
    "mpd.slice.971000-971999.json",
    "mpd.slice.972000-972999.json",
    "mpd.slice.973000-973999.json",
    "mpd.slice.974000-974999.json",
    "mpd.slice.975000-975999.json",
    "mpd.slice.976000-976999.json",
    "mpd.slice.977000-977999.json",
    "mpd.slice.978000-978999.json",
    "mpd.slice.979000-979999.json",
    "mpd.slice.98000-98999.json",
    "mpd.slice.980000-980999.json",
    "mpd.slice.981000-981999.json",
    "mpd.slice.982000-982999.json",
    "mpd.slice.983000-983999.json",
    "mpd.slice.984000-984999.json",
    "mpd.slice.985000-985999.json",
    "mpd.slice.986000-986999.json",
    "mpd.slice.987000-987999.json",
    "mpd.slice.988000-988999.json",
    "mpd.slice.989000-989999.json",
    "mpd.slice.99000-99999.json",
    "mpd.slice.990000-990999.json",
    "mpd.slice.991000-991999.json",
    "mpd.slice.992000-992999.json",
    "mpd.slice.993000-993999.json",
    "mpd.slice.994000-994999.json",
    "mpd.slice.995000-995999.json",
    "mpd.slice.996000-996999.json",
    "mpd.slice.997000-997999.json",
    "mpd.slice.998000-998999.json",
    "mpd.slice.999000-999999.json",
]


def make_test_data(data_dir, mv):
    all_files = os.listdir(data_dir)
    test_data_name = "mpd.slice.{x}00000-{x}00999.json"
    test_files = [test_data_name.format(x=i) for i in range(1, 10)]
    test_files.append("mpd.slice.0-999.json")
    assert set(test_files).issubset(all_files)
    test_json = {"info": [], "playlists": []}
    for f in test_files:
        with open(os.path.join(data_dir, f)) as fh:
            curr_json = json.load(fh)
            test_json["info"].append(curr_json["info"])
            test_json["playlists"].extend(curr_json["playlists"])
    out_dir = "test_data"
    filename = "mpd.test.json"
    make_output_file(test_json, test_files, out_dir, filename)


def make_validation_data(data_dir, mv):
    all_files = sorted(os.listdir(data_dir))
    # Use the 50 files (50k playlists) for validation
    validation_files = VALIDATION_FILES
    validation_json = {"info": [], "playlists": []}
    for f in validation_files:
        with open(os.path.join(data_dir, f)) as fh:
            curr_json = json.load(fh)
            validation_json["info"].append(curr_json["info"])
            validation_json["playlists"].extend(curr_json["playlists"])
    out_dir = "validation_data"
    filename = "mpd.validation.json"
    make_output_file(validation_json, validation_files, out_dir, filename)


def make_output_file(json_list, file_list, out_dir, filename):
    # shuffle playlists for good measure
    # note that this does not actually shuffle the order
    # of each playlist's tracks, which is important to preserve.
    random.shuffle(json_list["playlists"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w+") as fh:
        json.dump(json_list, fh, indent=4)
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
