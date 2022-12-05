# Diarization of audio files with Pyannote

## Running diarization

To run diarization, use the script `diarize.py`.
This script takes 3-4 arguments. 

The first argument is an access token that you will need to generate on huggingface by following the information here: https://huggingface.co/docs/hub/security-tokens. 
The Pyannote pipeline used in `diarize.py` requires that you provide your information to the creators in order to download and use their model on this page: https://huggingface.co/pyannote/speaker-diarization. 
Once you've gotten access to the trained model and generated your access token, you can copy it to your clipboard and paste it directly into your terminal as the first argument here.

The second argument is the directory containing one or more audio files to diarize. 
The third argument is a csv file containing a single column with the names of the audio files to diarize. 
Each audio file from the directory that is listed in the csv file is diarized, and the results are saved either in a custom directory specified by an optional third argument or in a directory called `diarized_files` that is in the same base directory as the directory containing audio files. 

The fourth optional argument is the location where you want `.rttm` diarized files to be saved.
If you do not provide this argument, `diarize.py` assumes you have a folder one directory up from the subdirectory containing audio files that is called `diarized_files`, and will save it there. 
Please ensure that you have created a folder called `diarized_files` in the appropriate location if you choose not to include this fourth argumment.

From the base directory of this repo:
```
python diarization/scripts/diarize.py path/to/audio path/to/filelist.csv
```