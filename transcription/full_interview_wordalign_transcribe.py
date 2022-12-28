# use whisperx for forced word alignment of transcription
import whisperx

device = "cpu"
audio_file = "data/RE19203c1fd13d1109b4db124da1210245.wav"

# transcribe with original whisper
model = whisperx.load_model("large", device)
result = model.transcribe(audio_file, fp16=False)

# load alignment model and metadata
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# align whisper output
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, device)

print(result["segments"]) # before alignment

print(result_aligned["segments"]) # after alignment
print(result_aligned["word_segments"]) # after alignment

with open('output/test_wd_aligned.csv', 'w') as testf:
    testf.write(result_aligned["word_segments"])