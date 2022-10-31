import whisper
import os
import numpy as np
import torch
import pandas as pd

# Whisper runs quicker with GPU. We transcribed a podcast of 1h and 10 minutes with Whisper.
# It took: 56 minutes to run it with GPU on local machine and 4 minutes to run it ith GPU on cloud environemnt

torch.cuda.is_available()
Device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("base", device=Device)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'}"
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)


turn_diarisation = pd.read_csv("data/turn_diarisation.csv")
turn_diarisation["split_wav_file_path"] = None
for i in range(len(turn_diarisation)):
    turn_diarisation["split_wav_file_path"][i] = os.path.join("./split_audio",
                                                              turn_diarisation["split_wav_file_name"][i])
print(turn_diarisation)

# Transcribe the entire audio file with transcribe command and print the results
interview = []
transcribe = []
transcribe_segment = []
language = []
for i in range(len(turn_diarisation)):
    if turn_diarisation["turn_length"][i] > 0:
        transcribe_result = model.transcribe(turn_diarisation["split_wav_file_path"][i],
                                         language=turn_diarisation["language"][i], fp16=False)
        transcribe_segment.append(transcribe_result['segments'])
        interview.append(turn_diarisation["split_wav_file_name"][i])
        language.append(turn_diarisation["language"][i])
        transcribe.append(transcribe_result["text"])

transcribe_time_length = []
for idx, item in enumerate(transcribe_segment):
    sum = 0
    for i in range(len(item)):
        length = item[i]['end'] - item[i]['start']
        sum += length
    transcribe_time_length.append(sum)
print('transcibe_time_length', transcribe_time_length)

import pandas as pd
data = pd.DataFrame(list(zip(interview, language, transcribe, transcribe_time_length)))
data.columns = ['split_wav_file_name','language','transcription','transcribe_time_length']
print(data)
data.to_csv("./output/split_audio_transcription.csv", index=False)

