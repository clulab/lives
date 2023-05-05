import argparse
import json
import os.path
import whisper


# ugly hack for ssl.SSLCertVerificationError
# https://stackoverflow.com/a/28052583/384641
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def construct_dataset(project_json_path, wav_file_dir):
    model = whisper.load_model("large-v2")

    with open(project_json_path) as json_file:
        project_json = json.load(json_file)
        for file_json in project_json:
            wav_file_name = file_json["file_upload"]
            print(wav_file_name)
            wav_path = os.path.join(wav_file_dir, wav_file_name)
            audio = whisper.load_audio(wav_path)
            for annotations_json in file_json["annotations"]:
                for result_json in annotations_json["result"]:
                    if result_json["type"] == "choices":
                        turn_start = result_json["value"]["start"]
                        turn_end = result_json["value"]["end"]
                        labels = result_json["value"]["choices"]

                        turn_start = int(turn_start * whisper.audio.SAMPLE_RATE)
                        turn_end = int(turn_end * whisper.audio.SAMPLE_RATE)
                        turn_audio = audio[turn_start:turn_end]
                        whisper_json = whisper.transcribe(model, turn_audio)
                        text = whisper_json["text"]

                        print(f"{labels}\n{text}")


def train():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.set_defaults(func=construct_dataset)
    dataset_parser.add_argument("project_json_path")
    dataset_parser.add_argument("wav_file_dir")

    kwargs = vars(parser.parse_args())
    kwargs.pop("func")(**kwargs)
