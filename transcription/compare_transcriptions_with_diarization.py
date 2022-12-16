# compare the turns marked by whisper with the diarization endpoints

import pandas as pd
from pathlib import Path


def compare_multiple_files(trans, diarized):
    """
    Compare timestamps on multiple files
    """
    all_files = diarized.fname.unique()

    all_results = {}
    for item in all_files:
        file_trans = trans[trans.wav_file_name == item]
        file_diar = diarized[diarized.fname == item]

        file_crossed_bounds = compare_timestamps(file_trans, file_diar)

        all_results[item] = {"crossed": sum(file_crossed_bounds),
                             "total": len(file_crossed_bounds),
                             "perc_crossed": sum(file_crossed_bounds) / float(len(file_crossed_bounds)),
                             "items": file_crossed_bounds}

    return all_results


def compare_timestamps(transcription, diarization):
    """
    Compare timestamps for transcription with those of diarization
    Specifically, we want to determine where transcriptions
    cross diarization turn boundaries
    """
    turn_boundaries = []
    # get all turn boundary times from diarization
    last_end = 0.0
    for i, row in diarization.iterrows():
        if i == 0:
            last_end = row.turn_end
        if i > 0:
            if row.turn_start > last_end:
                turn_boundaries.append((last_end, row.turn_start))
            if row.turn_end > last_end:
                last_end = row.turn_end

    # do another for loop to check where each turn starts/ends
    crossed_bounds = []
    for i, row in transcription.iterrows():
        for j, item in enumerate(turn_boundaries):
            if row.turn_start > item[1]:
                turn_boundaries.pop(j)
            elif row.turn_end > item[1]:
                crossed_bounds.append(1)
                break
            else:
                crossed_bounds.append(0)
                break

    return crossed_bounds


if __name__ == "__main__":
    # read in the transcriptions
    transcriptions = "output/full_test_transcription2.csv"
    tr = pd.read_csv(transcriptions)

    # read in the turn level audio file
    split_path = "/media/jculnan/datadrive/lives_data_copy/diarized_csv/test_ten.csv"
    di = pd.read_csv(split_path)

    # compare times
    results = compare_multiple_files(tr, di)

    total = 0
    total_crossed = 0

    for k, v in results.items():
        print(f"Results for item {k}: {str(v['perc_crossed'])}")
        total += v['total']
        total_crossed += v['crossed']

    print("Overall results for these files:")
    print(total_crossed/float(total))