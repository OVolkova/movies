import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from html2text import html2text


@dataclass
class Scene:
    text: str
    start: float
    end: float
    n_lines: int
    movie_id: str
    text_len: int


class ProcessMovieSubtitles:
    """
    Process full subtitles file into scenes
    """

    def __init__(
        self,
        movie_id: str,
        folder_path: Path = Path(os.path.expandvars("$MOVIES_DATA/splits")),
        scenes_cut_quantile: float = 0.95,
        min_scene: int = 300,
    ):
        """
        :param movie_id: id of movie to read subtitles
        :param folder_path: folder path where subtitles are stored
        :param scenes_cut_quantile: quantile for delta between
                two lines to cut into different scenes
        :param min_scene: minimum text length of one scene
        """
        self.movie_id = movie_id
        self.subs = pd.read_csv(Path(folder_path, movie_id + ".csv"))
        self._quantile = scenes_cut_quantile
        self._min_scene = min_scene

    def preprocess_texts(self):
        """
        convert html to text, replace special symbols
        """
        self.subs["text"] = self.subs["text"].apply(html2text)
        self.subs["text"] = self.subs["text"].str.strip().str.replace("\t", " ")
        self.subs["text"] = self.subs["text"].str.strip().str.replace("\n", " ")
        self.subs["text"] = (
            self.subs["text"].str.strip().str.replace(r"\\", "", regex=True)
        )
        self.subs["text"] = self.subs["text"].str.strip().str.replace("-", "")

    def preprocess_timeline(self):
        """
        calculate:
         - delta between subtitles lines,
         - duration of one line
         - text length and speed in one line
        This information is used to combine subtitles lines into scenes
        """

        def _clean_times(s: np.array, e: np.array) -> Tuple[np.array, np.array]:
            """
            s: start time
            e: end time
            d: duration
            """
            d = e - s
            e = np.where(d < 0, 1, 0) * s + np.where(d >= 0, 1, 0) * e
            d = e - s
            return e, d

        self.subs["end_time"], self.subs["duration"] = _clean_times(
            self.subs["start_time"].values, self.subs["end_time"].values
        )
        self.subs["delta"] = 0
        self.subs.loc[1:, "delta"] = (
            self.subs.loc[1:, "start_time"].values
            - self.subs.loc[: len(self.subs) - 2, "end_time"].values
        )
        self.subs["text_len"] = self.subs["text"].str.len()
        self.subs["text_speed"] = np.divide(
            self.subs["text_len"].astype(float).values,
            self.subs["duration"].values,
            out=np.zeros_like(self.subs["text_len"].astype(float).values),
            where=self.subs["duration"].values != 0,
        )

    def prepare_scenes(self) -> List[Scene]:
        """
        Combine subtitles lines into scenes. Two lines will be split into different scenes if
         - delta in time between lines is longer than self._quantile of distribution of
            all deltas
         - current scene text length is greater than minimum set value(self._min_scene)
        :return: List is scenes with its texts
        """
        value = self.subs["delta"].quantile(self._quantile)
        scenes = []
        scene_start = 0
        scene = ""
        i = None
        for i, (text, delta) in enumerate(self.subs[["text", "delta"]].values):
            if delta > value and len(scene) > self._min_scene:
                scenes.append(
                    Scene(
                        text=scene.strip(),
                        start=self.subs.at[scene_start, "start_time"],
                        end=self.subs.at[i - 1, "start_time"],
                        n_lines=i - scene_start,
                        movie_id=self.movie_id,
                        text_len=len(scene.strip()),
                    )
                )
                scene_start = i
                scene = ""
            scene += text
            scene += " "
        if i is None:
            return []
        if len(scene) > self._min_scene or len(scenes) == 0:
            scenes.append(
                Scene(
                    text=scene.strip(),
                    start=self.subs.at[scene_start, "start_time"],
                    end=self.subs.at[i, "start_time"],
                    n_lines=i - scene_start,
                    movie_id=self.movie_id,
                    text_len=len(scene.strip()),
                )
            )
        else:
            scenes[-1] = Scene(
                text=scenes[-1].text + " " + scene.strip(),
                start=scenes[-1].start,
                end=self.subs.at[i, "start_time"],
                n_lines=i - scene_start + scenes[-1].n_lines,
                movie_id=self.movie_id,
                text_len=len(scenes[-1].text + " " + scene.strip()),
            )
        return scenes


def main(args):
    """
    :param args: movie id and folder path where subtitles are stored
    :return: List of scenes of the file
    """
    movie_id, path = args
    subs = ProcessMovieSubtitles(movie_id, path)
    subs.preprocess_texts()
    subs.preprocess_timeline()
    return subs.prepare_scenes()


if __name__ == "__main__":
    import multiprocessing as mp
    from itertools import chain

    from tqdm.contrib.concurrent import process_map

    cpu_count = mp.cpu_count()
    splits_path = Path(os.path.expandvars("$MOVIES_DATA/splits"))

    movies_scenes = process_map(
        main,
        [(Path(file_name).stem, splits_path) for file_name in os.listdir(splits_path)],
        chunksize=1,
        max_workers=cpu_count // 2,
    )
    pd.DataFrame(chain.from_iterable(movies_scenes)).to_csv(
        os.path.expandvars("$MOVIES_DATA/scenes/scenes.csv"), sep="\t"
    )
