# Diarization of audio files with Pyannote

## Running diarization

To run diarization, use the script `diarize.py`. 

This script takes 2 arguments. 
The first argument is the directory containing one or more audio files to diarize. 
The second is a csv file containing a single column with the names of the audio files to diarize. 
Each audio file from the directory that is listed in the csv file is diarized, and the results are saved either in a custom directory specified by an optional third argument or in a directory called `diarized_files` that is in the same base directory as the directory containing audio files. 

From the base directory of this repo:
```
python diarization/scripts/diarize.py path/to/audio path/to/filelist.csv
```