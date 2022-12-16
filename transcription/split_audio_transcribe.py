import pandas as pd
from pathlib import Path
import sys
sys.path.append("/home/jculnan/github/lives")  # change or remove, as needed
from transcription.whisper_utils import set_device, load_model, save_transcriptions


class SplitAudioTranscriber:
    def __init__(self, diarized_df):
        self.diarized = diarized_df

        self.device = set_device()
        self.model = load_model(self.device)

    def transcribe_audio_clips(self):
        """
        Transcribe all audio clips listed in a diarized df
        """
        # set holders for output
        clip_names = []
        transcriptions = []
        segments = []
        clip_languages = []

        # for turn in diarized df
        for i in range(len(self.diarized)):
            # if turn has duration greater than 0
            if self.diarized["turn_length"][i] > 0:
                # transcribe the turn
                transcribe_result = self.model.transcribe(self.diarized["split_wav_file_path"][i],
                                                          language=self.diarized["language"][i],
                                                          fp16=False)
                # add the relevant components of the transcription to holders
                segments.append(transcribe_result['segments'])
                transcriptions.append(transcribe_result["text"])

                # add the name of the audio clip and its language to holders
                clip_names.append(self.diarized["split_wav_file_name"][i])
                clip_languages.append(self.diarized["language"][i])

        # get length of each transcribed segment
        len_transcriptions = []

        for idx, item in enumerate(segments):
            sum = 0
            for i in range(len(item)):
                length = item[i]['end'] - item[i]['start']
                sum += length
            len_transcriptions.append(sum)

        data = pd.DataFrame(list(zip(clip_names, clip_languages, transcriptions, len_transcriptions)))
        data.columns = ['split_wav_file_name', 'language', 'transcription', 'transcription_length']

        return data


def get_diarized_df(diarized_csv_path, path_to_split_audio):
    """
    Prepare the diarized df from diarization csv file
    :param diarized_csv_path: a string for the diarization csv
    :param path_to_split_audio: a Path to the split audio files
    """
    diarized = pd.read_csv(diarized_csv_path)

    fnames = diarized.fname.str.split(".wav").str[0]
    starts = round(diarized.turn_start, 3).astype(str)
    ends = round(diarized.turn_end, 3).astype(str)

    diarized["split_wav_file_name"] = fnames + "_" + starts + "-" + ends + ".wav"

    diarized["split_wav_file_path"] = ''
    if "language" not in diarized.columns:
        diarized["language"] = "english"

    for i in range(len(diarized)):
        diarized["split_wav_file_path"][i] = \
            str(path_to_split_audio / f"{fnames[i]}/{diarized.split_wav_file_name[i]}")

    return diarized


if __name__ == "__main__":
    # change paths as necessary
    diarized_path = "/media/jculnan/datadrive/lives_data_copy/diarized_csv/test_ten.csv"
    split_path = Path("/media/jculnan/datadrive/lives_data_copy/split_audio/test_ten")

    df = get_diarized_df(diarized_path, split_path)

    transcriber = SplitAudioTranscriber(df)
    data = transcriber.transcribe_audio_clips()
    save_transcriptions(data, "split_test_transcription.csv")