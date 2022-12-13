# calculate the places where pyannote says there's no speech
# but the gold rttm files says there IS speech

from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
import subprocess as sp


def subset_diff_rttm(diff_rttm):
    """
    Get the subset of the diff rttm
    Have to do this as pandas df as the rttm loader
    from pyannote does not read this in correctly
    """
    diff = pd.read_csv(diff_rttm,
                       header=None,
                       names=["na0",
                              "file",
                              "num",
                              "start_time",
                              "dur",
                              "na1",
                              "na2",
                              "speaker",
                              "na3",
                              "na4"],
                       sep=" ")

    diff = diff[diff["speaker"][0] == "missed detection"]

    return diff


class CombinedMissed:
    def __init__(self, missed_dir):
        self.missed = Path(missed_dir) if type(missed_dir) == str else missed_dir

        self.combined = self.combine_missed_files()

    def combine_missed_files(self):
        """
        Combine all the files containing missed detection
        """
        all_missed = None
        for f in self.missed.iterdir():
            if f.suffix == ".csv":
                missed = pd.read_csv(f)
                missed['fname'] = f.stem.split("_")[0]

                if all_missed is None:
                    all_missed = missed
                else:
                    all_missed = all_missed.append(missed)

        return all_missed

    def sample_from_missed(self, sample_size=500, lower=0.0, upper=1000.0):
        """
        Sample randomly from within the df of all missed
        :param all_missed: pandas df of all missed items
        :param sample_size: number to sample
        :param lower: lower bound on length of time for sample
        :param upper: upper bound on length of time for sample
        :returns: df of samples

        Bounds set to allow for a sample from items within
        0.5 and 2 seconds, for example, and add to these all
        items over 2 seconds later
        """
        all_missed_bounded = self.get_all_samples_of_size(lower, upper)
        all_missed_bounded = all_missed_bounded.sample(n=sample_size)

        return all_missed_bounded

    def get_all_samples_of_size(self, lower=0.0, upper=1000.0):
        """
        Get all items that fall within the specified bounds
        """
        return self.combined[self.combined["length"].between(lower, upper)]

    def save_combined_missed_files(self):
        self.combined.to_csv(self.missed / "all_missed.csv", index=False)


def split_wav_of_samples(wav_path, samples_df, save_path=None):
    """
    Segment out the portions of wav files that were missed
    So that each may be listened to in isolation
    """
    if save_path is None:
        save_path = wav_path / "missed_samples"
    os.makedirs(save_path, exist_ok=True)

    print(save_path)

    fileset = samples_df['fname'].unique().tolist()
    print(fileset)

    available_calls = [item.name for item in wav_path.iterdir() if item.is_file()]

    for item in fileset:
        item_wav = item + ".wav"
        if item_wav in available_calls:
            print(item_wav)
            this_wav = wav_path / item_wav
            to_cut = samples_df[samples_df['fname'] == item]

            for row in tqdm(to_cut.itertuples(), total=len(to_cut)):
                id = row.fname
                timestart = row.start
                timeend = row.end

                sp.run(
                    [
                        "ffmpeg",
                        "-i",
                        str(this_wav),
                        "-ss",
                        str(timestart),
                        "-to",
                        str(timeend),
                        f"{str(save_path)}/{id}_start{str(timestart)}_end{str(timeend)}.wav",
                        "-loglevel",
                        "quiet",
                    ]
                )



# get set of files


# for f in set of files:
    # if found in directory:
        # subset of items from this file
        #

class AudioSplit:
    """ Takes audio, can split using ffmpeg"""
    def __init__(self, base_path, audio_name, save_ext=None):
        self.path = base_path,
        self.fname = audio_name
        if not audio_name.endswith(".wav"):
            self.fpath = f"{base_path}/{audio_name}.wav"
        else:
            self.fpath = f"{base_path}/{audio_name}"

        if save_ext is not None:
            self.savepath = f"{base_path}/{save_ext}"
        else:
            self.savepath = base_path

    def split_audio_with_pandas(self, utt_df):
        """
        Split audio file based on input pandas df
        Df contains columns ['speaker'], ['start'], ['end']
        :return:
        """

        os.makedirs(self.savepath, exist_ok=True)

        for row in tqdm(utt_df.itertuples(), total=len(utt_df)):
            # print(row)
            recording_id = row.fname
            timestart = row.start
            timeend = row.end

            sp.run(
                [
                    "ffmpeg",
                    "-i",
                    self.fpath,
                    "-ss",
                    str(timestart),
                    "-to",
                    str(timeend),
                    f"{self.savepath}/{recording_id}_start{str(timestart)}_end{str(timeend)}.wav",
                    "-loglevel",
                    "quiet",
                ]
            )



if __name__ == "__main__":
    base_path = "/media/jculnan/backup/From LIVES folder/"
    annotated = f"{base_path}/diarized_files"

    # missed = CombinedMissed(annotated)
    # missed.save_combined_missed_files()
    #
    # biggest = missed.get_all_samples_of_size(lower=2.0)
    # smaller_sample = missed.sample_from_missed(sample_size=500, lower=0.250, upper=1.999)
    #
    # samples = biggest.append(smaller_sample)
    # samples.to_csv(f"{annotated}/sampled_missed_items.csv", index=False)

    # after running the above
    samples = pd.read_csv(f"{annotated}/sampled_missed_items.csv")
    audio_1 = Path("/media/jculnan/backup/pull-twilio-recording-script-data")
    # audio_2 = Path("/media/jculnan/backup/From LIVES folder/Spanish_calls")
    save_path = Path("/media/jculnan/backup/From LIVES folder/diarization_effort/english_audio")

    split_wav_of_samples(audio_1, samples, save_path)