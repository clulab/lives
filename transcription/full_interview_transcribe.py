import whisper
import pandas as pd
from pathlib import Path
import sys
sys.path.append("/home/jculnan/github/lives")
print(sys.path)
from transcription.whisper_utils import set_device, load_model, save_transcriptions

# Whisper runs quicker with GPU. We transcribed a podcast of 1h and 10 minutes with Whisper.
# It took: 56 minutes to run it with GPU on local machine and 4 minutes to run it ith GPU on cloud environemnt


class AudioTranscriber:
    def __init__(self, audio_files_pathstr, languages=None):
        # param languages is either a list or None
        self.device = set_device()
        self.model = load_model(self.device)
        self.audio_path = Path(audio_files_pathstr)

        self.all_audio_paths, self.all_audio_names = self._get_all_audio_names()

        self.language_preds = None
        if languages is None:
            self.language = "english"
        elif len(languages) > 1:
            # get language predictions for each moment in time
            self.language_preds = self._get_language_predictions()
        else:
            # if only one language, no need for this step
            self.language = languages[0].lower()

    def _get_all_audio_names(self):

        all_audio_paths = []
        file_names = []

        for fname in self.audio_path.iterdir():
            if fname.suffix == ".wav":
                all_audio_paths.append(str(fname))
                file_names.append(str(fname.name))

        return all_audio_paths, file_names

    def _get_language_predictions(self):

        # create holder for mel spectrograms
        Mel = []

        for i in range(len(self.all_audio_paths)):
            # Open an audio file and read as mono waveform
            audio = whisper.load_audio(self.all_audio_paths[i])

            # Pad or trim the audio aray to N_SAMPLES, as expected by the encoder
            audio = whisper.pad_or_trim(audio)

            # Compute the log-Mel spectrogram of
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            Mel.append(mel)

        lang_preds = []
        # Detect_language detects the audio-file language:
        for m in Mel:
            # Return them as list of strings, along with the ids of the most probable language tokens - "_"
            # and the probability distribution over all language tokens - "probs.
            _, probs = self.model.detect_language(m)
            lang_preds.append(max(probs, key = probs.get))

        return lang_preds

    def transcribe_audio_files(self):
        # Transcribe the entire audio file with transcribe command and print the results
        # create holders for data
        clip_names = []
        transcriptions = []
        clip_languages = []
        segments = []

        # iterate over files to transcribe and transcribe them
        for i in range(len(self.all_audio_paths)):
            # check whether we got predictions of language for each part
            if self.language_preds is not None:
                result = self.model.transcribe(self.all_audio_paths[i],
                                               language=self.language_preds[i],
                                               fp16=False)
            # else, if we know we are only using one language
            else:
                result = self.model.transcribe(self.all_audio_paths[i],
                                               language=self.language,
                                               fp16=False)

            # add info from transcription to holders
            segments.append(result['segments'])
            transcriptions.append(result["text"])

            # add name of clip and its language to file
            clip_names.append(self.all_audio_names[i])
            clip_languages.append(self.language_preds[i] if self.language_preds else self.language)

        # calculate the length of time that was transcribed
        len_transcriptions = []
        for idx, item in enumerate(segments):
            sum = 0
            for i in range(len(item)):
                length = item[i]['end'] - item[i]['start']
                sum += length
            len_transcriptions.append(sum)

        data = pd.DataFrame(list(zip(clip_names, clip_languages, transcriptions, len_transcriptions)))
        data.columns = ['wav_file_name', 'language', 'transcription', 'transcibe_time_length']

        return data


if __name__ == "__main__":
    audio_path = Path("/home/jculnan/github/lives/data")

    transcriber = AudioTranscriber(audio_path, languages=["english"])
    data = transcriber.transcribe_audio_files()
    save_transcriptions(data, "full_test_transcription.csv")


