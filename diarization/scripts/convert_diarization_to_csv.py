# convert a pyannote .rttm file to a .csv file to use with whisper
# this script should run over multiple files at once if needed
# all turns for one file will be given in one .csv

from pathlib import Path
import pandas as pd
import subprocess as sp
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
        self.savepath = savepath_str
        self.o_path = Path(savepath_str)
        # create directories in o_path if they don't yet exist
        self.o_path.mkdir(parents=True, exist_ok=True)

        self.dfs_dict = self.read_in_rttm()

    def read_in_rttm(self):
        # read in all rttms
        # save in dict
        dfs_dict = {}

        for f in self.i_path.iterdir():
            if f.suffix == ".rttm":
                rttm_df = pd.read_csv(f,
                                      names=["p1", "fname",
                                             "p2", "turn_start",
                                             "turn_length", "p3",
                                             "p4", "speaker",
                                             "p5", "p6"], sep=" ")

                # subset only necessary columns
                rttm_df = rttm_df[["fname", "turn_start", "turn_length", "speaker"]]

                # add .wav to end of fname for use with whisper scripts
                rttm_df.fname = rttm_df.fname.apply(lambda x: x + ".wav")

                # calculate turn end from length of turn
                rttm_df['turn_end'] = rttm_df['turn_start'] + rttm_df['turn_length']

                # calculate time between turns
                rttm_df['time_to_next'] = rttm_df["turn_start"].shift(-1) - rttm_df["turn_end"]

                # todo: ID turns that are close together and by same speaker
                #   and collapse over these

                dfs_dict[f.stem] = rttm_df

        return dfs_dict

    def save_csv_files(self, together=False):
        # save the csv files in the dfs dict -- either all separately or as a single file
        # :param together: whether to save all dfs together
        if not together:
            for k, v in self.dfs_dict.items():
                v.to_csv(f"{self.savepath}/{k}.csv", index=False)
        else:
            all_dfs_list = [df for _, df in self.dfs_dict.items()]
            all_dfs = pd.concat(all_dfs_list)
            all_dfs.to_csv(f"{self.savepath}/all_files_together.csv", index=False)


    def chunk_wav_files(self, wav_read_pathstr, wav_save_pathstr):
        # create output directory
        wav_save_path = Path(wav_save_pathstr)
        wav_save_path.mkdir(parents=True, exist_ok=True)

        for item in self.dfs_dict.keys():
            # create subdir for this wav file
            itemname = item.split(".wav")[0]
            itempath = wav_save_path / itemname
            itempath.mkdir(exist_ok=True)

            # get the relevant df out of df dict
            df = self.dfs_dict[item]

            # for each line in this df:
            for row in df.itertuples():

                # chunk using start and end times in df
                sp.run(["ffmpeg","-ss", str(row.turn_start),  "-i", f"{wav_read_pathstr}/{row.fname}",
                        "-t", str(row.turn_length), "-c", "copy",
                        f"{str(itempath)}/{itemname}_{str(round(row.turn_start, 3))}-{str(round(row.turn_end, 3))}.wav"])


if __name__ == "__main__":
    rttmpath = "/media/jculnan/datadrive/lives_data_copy/diarized_files"
    savepath = "/media/jculnan/datadrive/lives_data_copy/diarized_csv"

    x = RTTMConverter(rttmpath, savepath)
    x.read_in_rttm()
    x.chunk_wav_files(
        wav_read_pathstr="/media/jculnan/datadrive/lives_data_copy/Call recordings, all fidelity scored calls, n=323",
        wav_save_pathstr="/media/jculnan/datadrive/lives_data_copy/split_audio"
    )
    x.save_csv_files(together=True)
