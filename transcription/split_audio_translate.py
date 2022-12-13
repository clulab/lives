import transcription
import os
import numpy as np
import torch
import pandas as pd

# Whisper runs quicker with GPU. We transcribed a podcast of 1h and 10 minutes with Whisper.
# It took: 56 minutes to run it with GPU on local machine and 4 minutes to run it ith GPU on cloud environemnt

torch.cuda.is_available()
Device = "cuda" if torch.cuda.is_available() else "cpu"

model = transcription.load_model("base", device=Device)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'}"
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

audio_name = []
file = []
lang = []
turn_diarisation = pd.read_csv("data/turn_diarisation.csv")
for i in range(len(turn_diarisation)):
    if turn_diarisation["turn_length"][i] > 0 and turn_diarisation["language"][i] == "es":
        file.append(turn_diarisation["split_wav_file_name"][i])
        audio_name.append(os.path.join("./split_audio", turn_diarisation["split_wav_file_name"][i]))
        lang.append(turn_diarisation["language"][i])
print(len(file))
print(len(audio_name))
print(len(lang))
print(lang)

# Transcribe the entire audio file with transcribe command and print the results
interview = []
translate = []
translate_segment = []
for i in range(len(audio_name)):
    translate_result = model.transcribe(audio_name[i], language="en", fp16=False)
    translate_segment.append(translate_result['segments'])
    interview.append(file[i])
    translate.append(translate_result["text"])

translate_time_length = []
for idx, item in enumerate(translate_segment):
    sum = 0
    for i in range(len(item)):
        length = item[i]['end'] - item[i]['start']
        sum += length
    translate_time_length.append(sum)
print('translation_time_length', translate_time_length)

import pandas as pd
data = pd.DataFrame(list(zip(interview, lang, translate, translate_time_length)))
data.columns = ['wav_file_name','language','transcription','translate_time_length']
print(data)
data.to_csv("./output/split_audio_translation.csv", index=False)

