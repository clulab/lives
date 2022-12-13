import sys
# fixme
sys.path.append("/home/jculnan/github/lives")
from diarization.pyannote_diarization import diarize_n_files
from pyannote.audio import Pipeline
import logging
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
        ]
    )

    if len(sys.argv) >= 4:
        # access token for huggingface
        ACCESS_TOKEN = sys.argv[1]
        # directory containing audio files
        AUDIO_DIR = sys.argv[2]

        # a csv file containing a list of the files to diarize
        # assumes list of files is the first column in this file
        # NOTE: do not include extensions in file names
        FILES_TO_DIARIZE = sys.argv[3]

        if len(sys.argv) > 4:
            SAVE_DIR = sys.argv[4]
        else:
            SAVE_DIR = f"{AUDIO_DIR}/../diarized_files"
    else:
        exit("Please specify a directory containing your audio file and a txt containing a list of files to diarize")

    # get files to diarize
    to_diarize = pd.read_csv(FILES_TO_DIARIZE)
    f_to_diarize = to_diarize.iloc[:, 0].tolist()

    # diarize the files
    # pyannote now requires you to sign in to use their models
    # create an access token on huggingface and use this here
    # you also need to give your information on each of the models used in the pipeline
    # you do NOT need to be affiliated with pyannote to do this
    # this ONLY works with pyannote.audio v 2.1+
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=ACCESS_TOKEN)
    diarize_n_files(pipeline, f_to_diarize, AUDIO_DIR, SAVE_DIR)