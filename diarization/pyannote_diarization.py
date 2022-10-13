# diarization with pyannote

# preparing notebook for visualization purposes
# (only show outputs between t=0s and t=30s)
from pyannote.core import notebook, Segment
from pyannote.database.util import load_rttm
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Pipeline

from tqdm import tqdm as tqdm

import pandas as pd
import os, sys
from pathlib import Path
import logging


def diarize_n_files(diarization_pipeline, f_to_diarize, audio_dir, n):
    """
    Diarize n files
    """
    num_diarized = 0

    for f in f_to_diarize:
        if num_diarized < n:
            fpath = f"{audio_dir}/{f}.wav"
            logging.info(f"Searching for: {fpath}")
            if Path(fpath).exists():
                logging.info(f"Now starting diarization for file {num_diarized + 1}: {f}")
                diarization = diarization_pipeline(fpath, num_speakers=2)

                with open(f"{FILENAME_DIR}/diarized_files/{f}.rttm", "w") as rttm:
                    diarization.write_rttm(rttm)
                logging.info(f"Saved RTTM file for {f}")

                num_diarized += 1
            else:
                logging.info(f"File not found: {f}")
        else:

            break


if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
        ]
    )

    ROOT_DIR = "/media/jculnan/backup/pull-twilio-recording-script-data"
    FILENAME_DIR = "/media/jculnan/backup/From LIVES folder"
    AUDIO_DIR = f"{FILENAME_DIR}/Spanish_calls"

    # get the file containing all files to diarize
    #to_diarize = pd.read_csv(f"{FILENAME_DIR}/all_diarisation_ids.txt")
    to_diarize = pd.read_csv(f"{FILENAME_DIR}/spanish_diarization_ids.txt")
    f_to_diarize = to_diarize.iloc[:, 0].tolist()
    print(f_to_diarize)

    # tokenizer = BertTokenizer.from_pretrained('path/to/vocab.txt',local_files_only=True)
    #   once this is trained, change to just using local files
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    diarize_n_files(pipeline, f_to_diarize, AUDIO_DIR, 100)

# clone pyannote-audio Github repository and update ROOT_DIR accordingly
#ROOT_DIR = "/home/jculnan/github/pyannote-audio"
#AUDIO_FILE = f"{ROOT_DIR}/tutorials/assets/sample.wav"

# AUDIO_FILE = f"{ROOT_DIR}/RE0a16de4ff6b87de139f7b9ff59532f17.wav"

# # class for lives diarization inheriting from SpeakerDiarization class
# class LIvESDiarization(SpeakerDiarization):
#     pass
#
# # diarize
# pipeline = SpeakerDiarization(embedding="pyannote/embedding")#embedding="speechbrain/spkrec-ecapa-voxceleb")
# diarization = pipeline(AUDIO_FILE, num_speakers=2)
#
# print(diarization.clustering)




# # apply pretrained pipeline
# diarization = pipeline(AUDIO_FILE, num_speakers=2)
#
# # dump the diarization output to disk using RTTM format
# with open("test-audio.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)


# # print the result
# for item in diarization.itertracks(yield_label=True):
#     # add to list
#
#     print(item)
#
# # convert speaker preds to torch.Tensor
#
# # get
#
#
# for turn, _, speaker in diarization.itertracks(yield_label=True):
#     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
