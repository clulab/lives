
import pandas as pd


def convert_pred_rttm(pred_rttm_path):
    """
    Convert a pred rttm file to match the gold file format
    Specifically: pred file marks pauses as turn boundaries,
    while gold file does not
    This collapses predicted turns into a single larger turn
    to match the gold file
    """
    # first portion is from
    # https://github.com/pyannote/pyannote-database/blob/develop/pyannote/database/util.py
    names = [
        "NA1",
        "uri",
        "NA2",
        "start",
        "duration",
        "NA3",
        "NA4",
        "speaker",
        "NA5",
        "NA6",
    ]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}

    preds = pd.read_csv(
        pred_rttm_path,
        names=names,
        dtype=dtype,
        delim_whitespace=True,
        keep_default_na=False,
    )


