"""
Split one dataset file into one movie subtitles files
"""
import os
from pathlib import Path

import pandas as pd
import tqdm


def split_files(file_path: Path, new_folder: Path):
    subs = pd.read_csv(file_path)
    subs.dropna(inplace=True)
    subs.drop_duplicates(["imdb_id", "start_time", "end_time", "text"], inplace=True)
    subs = subs.sort_values(["imdb_id", "start_time", "end_time"]).reset_index(
        drop=True
    )
    for i, group in tqdm.tqdm(subs.groupby("imdb_id")):
        group.reset_index(drop=True).to_csv(Path(new_folder, str(i) + ".csv"))


if __name__ == "__main__":
    origin_path = Path(os.path.expandvars("$MOVIES_DATA/origin/movies_subtitles.csv"))
    splits_path = Path(os.path.expandvars("$MOVIES_DATA/splits"))
    split_files(origin_path, splits_path)
