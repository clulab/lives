# convert a pyannote .rttm file to a .csv file to use with whisper
# this script should run over multiple files at once if needed
# all turns for one file will be given in one .csv

from pathlib import Path
import pandas as pd
# import os


class RTTMConverter:
    def __init__(self, path_str, savepath_str):
        """
        :param path_str: string for path to dir containing one or
            more .rttm files that have been annotated with pyannote
        :param savepath_str: string for path where csvs should be saved
        """
        # create input and output paths
        self.i_path = Path(path_str)
        self.o_path = Path(savepath_str)
        # create directories in o_path if they don't yet exist
        self.o_path.mkdir(parents=True, exist_ok=True)

    def read_in_rttm(self):
        # read in all rttms
        # save in dict
        dfs_dict = {}

        for f in self.i_path.iterdir():
            if f.suffix == ".rttm":
                rttm_df = pd.read_csv(f,
                                      colnames=["p1", "fname",
                                                "p2", "turn_start",
                                                "turn_end", "p3",
                                                "p4", "speaker",
                                                "p5", "p6"])
                # subset only necessary columns
                rttm_df = rttm_df[["fname", "turn_start", "turn_end", "speaker"]]

                # calculate time between turns
                rttm_df['time_since_last'] = rttm_df["turn_start"].shift(-1) - rttm_df["turn_end"]

                # ID turns that are close together and by same speaker
                # and collapse over these

                dfs_dict[f.stem] = rttm_df

        return dfs_dict

    # def save_to_
