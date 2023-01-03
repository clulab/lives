# combine transcriptions with diarization
# this requires the use of transcriptions from overall files where successful
# and one-off transcription of diarized segments in places where
# utterance boundaries in transcription have failed
import pandas as pd
from pathlib import Path
import subprocess as sp
import sys
sys.path.append("/home/jculnan/github/lives")  # change or remove, as needed

from transcription.whisper_utils import set_device, load_model, save_transcriptions


class TransDiarCombiner:
    # a class that combines transcriptions and diarization
    # and handles additional diarization, as needed
    def __init__(self, transcription_df, diarization_df):
        self.trans = transcription_df
        # partially-diarized df
        self.diar = self.combine_successful(diarization_df)

    def combine_successful(self, diarized_df):
        # combine successfully transcribed items with diarized turns

        # add transcription to the diarization df
        all_trans_fnames = self.trans.wav_file_name.unique().tolist()
        all_diar_names = diarized_df.fname.unique().tolist()
        all_fnames = set(all_trans_fnames) & set(all_diar_names)

        updated_diarization = None

        for fname in all_fnames:
            di_df = diarized_df[diarized_df.fname == fname].reset_index(drop=True)
            tr_df = self.trans[self.trans.wav_file_name == fname].reset_index(drop=True)
            uncrossed = get_transcription_from_diarization_single_file(di_df, tr_df)

            di_df = di_df.merge(uncrossed, on=['fname', 'turn_start', 'turn_end'], how='left')
            di_df.to_csv("output/noncrossed_added.csv", index=False)

            # concatenate this updated df with other updates
            if updated_diarization is None:
                updated_diarization = di_df
            else:
                updated_diarization = pd.concat([updated_diarization, di_df])

        return updated_diarization

    def generate_other_transcriptions(self, audio_path, save_chunks=False):
        # get transcriptions where missing in a pd.DataFrame
        # requires all relevant audio files to be in audio_path
        full_diar = chunk_and_transcribe(self.diar, audio_path, save_chunks)

        # update self.diar to have all transcriptions
        self.diar = full_diar


def chunk_and_transcribe(df, wav_read_pathstr, save_chunks=False):
    # set device and model
    device = set_device()
    model = load_model(device)

    wav_read_path = Path(wav_read_pathstr)

    # create subdir for short wav file
    itempath = wav_read_path / "chunked"
    itempath.mkdir(exist_ok=True)

    # set holder for new df
    updated = []

    # for each line in this df:
    for i, row in df.iterrows():
        if not pd.isna(row.transcription):
            updated.append(row.to_dict())
        else:
            # get the path/name of saved file
            itemname = row.fname.split(".wav")[0]
            short_wav = f"{str(itempath)}/{itemname}_{str(round(row.turn_start, 3))}-{str(round(row.turn_end, 3))}.wav"

            # chunk using start and end times in df
            if not Path.exists(Path(short_wav)):
                sp.run(["ffmpeg","-ss", str(row.turn_start),  "-i", f"{wav_read_pathstr}/{row.fname}",
                        "-t", str(row.turn_length), "-c", "copy",
                        short_wav])

            # transcribe this chunk
            transcribe_result = model.transcribe(short_wav,
                                                 language="english",  # todo: add flexibility
                                                 fp16=False)

            # add transcription to diarized file
            row.transcription = transcribe_result["text"]
            updated.append(row.to_dict())

            # remove the chunk
            if not save_chunks:
                sp.run(["rm", short_wav])

    return pd.DataFrame(updated)


def get_transcription_from_diarization_single_file(diarized_df, trans_df):
    # get all turn boundary times from diarization
    turn_boundaries = []

    # get holder for end of previous item
    last_end = 0.0

    # add start and end of each turn to holder
    # as well as end of prev turn and start of next turn
    # for comparison with transcription turns
    for i, row in diarized_df.iterrows():
        if i == 0 or row.turn_end > last_end:
            # this ignores smaller turns that overlap with larger turns
            if not i+1 == len(diarized_df):
                turn_boundaries.append((last_end,
                                        row.turn_start,
                                        row.turn_end,
                                        diarized_df.loc[i+1].turn_start))
            else:
                # treat start of next turn loosely if this is the last row
                # for this file
                turn_boundaries.append((last_end,
                                        row.turn_start,
                                        row.turn_end,
                                        row.turn_end + 100)) # some number added
            last_end = row.turn_end

    # find all non-crossing items
    # these are items that start AFTER the previous turn ends
    # and end BEFORE the next turn starts
    noncrossed = []
    temp_holder = []
    for i, row in trans_df.iterrows():
        # item contains (prev_end, start, end, next_start)
        for j, item in enumerate(turn_boundaries):
            # remove this from list of boundaries if start is after end of this
            if row.turn_start > item[2]:
                turn_boundaries.pop(j)
            # if starts after prev turn ends and ends before next turn starts
            # and turns contain an overlap
            elif item[0] < row.turn_start < item[2] and item[1] < row.turn_end < item[3]:
                row_dict = row.to_dict()
                row_dict['matched_turn_start'] = item[1]
                row_dict['matched_turn_end'] = item[2]
                # check if this is the same turn that was matched in last item
                if len(temp_holder) > 0:
                    if temp_holder[-1]['matched_turn_start'] == item[1]:
                        temp_holder[-1]['turn_end'] = row.turn_end
                        temp_holder[-1]['transcription'] += row_dict['transcription']
                        break
                    else:
                        noncrossed.extend(temp_holder)
                        temp_holder = []
                # add this to list of non-crossed items
                temp_holder.append(row_dict)
                break
            else:
                break

    # get only the relevant columns of noncrossed
    noncrossed = pd.DataFrame(noncrossed,
                              columns=['wav_file_name',
                                       'matched_turn_start',
                                       'matched_turn_end',
                                       'transcription'])
    noncrossed.rename(columns={'wav_file_name': 'fname',
                               'matched_turn_start': 'turn_start',
                               'matched_turn_end': 'turn_end'}, inplace=True)

    return noncrossed


if __name__ == "__main__":
    diarization_path = "/media/jculnan/datadrive/lives_data_copy/diarized_csv/all_files_together.csv"
    transcription_path = "output/full_transcription_next-data.csv"

    transcription = pd.read_csv(transcription_path)
    diarization = pd.read_csv(diarization_path)

    trans_di = TransDiarCombiner(transcription, diarization)

    trans_di.generate_other_transcriptions("/media/jculnan/datadrive/lives_data_copy/Call recordings, all fidelity scored calls, n=323",
                                           save_chunks=True)
    trans_di.diar.to_csv("output/transcription_diarization_combined.csv",index=False)
