import transcription
import os
import numpy as np
import torch

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
for filename in os.listdir("./data"):
    if filename.endswith(".wav"):
        audio_name.append(os.path.join("./data", filename))
        file.append(filename)
print(audio_name)
print(file)
#print(len(audio_name))

Mel = []
for i in range(len(audio_name)):
    # Open an audio file and read as mono waveform
    audio = transcription.load_audio(audio_name[i])
    print(audio)
    print(len(audio))
    # Pad or trim the audio aray to N_SAMPLES, as expected by the encoder
    audio = transcription.pad_or_trim(audio)
    print(audio)
    # Compute the log-Mel spectrogram of
    mel = transcription.log_mel_spectrogram(audio).to(model.device)
    Mel.append(mel)
    print(mel)
print('Mel', Mel)

lang = []
# Detect_language detects the audio-file language:
for m in Mel:
    # Return them as list of strings, along with the ids of the most probable language tokens - "_"
    # and the probability distribution over all language tokens - "probs.
    _, probs = model.detect_language(m)
    print(f"Detected language: {max(probs, key = probs.get)}")
    lang.append(max(probs, key = probs.get))
print(lang)

# Transcribe the audio using the DecodingOptions and the decode command.
# Print the first 30 seconds of the audio
#for l in range(len(lang)):
    #print(l)
    #options = transcription.DecodingOptions(language=lang[l], without_timestamps=True, fp16=False)
    #print(options)
    #result = transcription.decode(model, Mel[l], options)
    #print(result.text)

# Transcribe the entire audio file with transcribe command and print the results
interview = []
language = []
translate = []
translate_segment = []
for i in range(len(audio_name)):
    if lang[i] == 'es':
        translate_result = model.transcribe(audio_name[i], language="en", fp16=False)
        translate_segment.append(translate_result['segments'])
        interview.append(file[i])
        language.append(lang[i])
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
data = pd.DataFrame(list(zip(interview, language, translate, translate_time_length)))
data.columns = ['wav_file_name','language','translation','translation_time_length']
print(data)
data.to_csv("./output/translation.csv")

