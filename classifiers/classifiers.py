import argparse
import collections
import json
import os.path
import datasets
import whisper


# ugly hack for ssl.SSLCertVerificationError
# https://stackoverflow.com/a/28052583/384641
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def construct_dataset(project_json_path, wav_file_dir, dataset_path):

    # gather LabelStudio annotations, grouping by (start, end) in the audio
    file_annotations = collections.defaultdict(dict)
    with open(project_json_path) as json_file:
        for file_json in json.load(json_file):
            wav_file_name = file_json["file_upload"]

            # each annotation is a LabelStudio "result"
            for annotations_json in file_json["annotations"]:

                # for now, use only Grey's annotations
                annotator = annotations_json["completed_by"]["email"]
                if annotator != "sarahjwright@email.arizona.edu":
                    continue

                for result_json in annotations_json["result"]:
                    value_json = result_json["value"]

                    # use all annotations with spans
                    if "start" in value_json:
                        turn_start = value_json["start"]
                        turn_end = value_json["end"]

                        match result_json["from_name"]:
                            case "labels":
                                labels = value_json["labels"]
                            case "coach_fidelity" |\
                                 "coach_techniques_adherence" |\
                                 "coach_constructs_good" |\
                                 "coach_constructs_bad" |\
                                 "coach_techniques_other" |\
                                 "coach_annotator_notes" |\
                                 "participant_mi_related" |\
                                 "participant_lives_goals" |\
                                 "participant_psychological_symptoms" |\
                                 "participant_annotator_notes":
                                labels = value_json["choices"]
                            case "coach_text_note" | "participant_text_note":
                                pass
                            case _:
                                raise NotImplementedError(str(result_json))

                        # save span and labels
                        span = turn_start, turn_end
                        if span not in file_annotations:
                            file_annotations[wav_file_name][span] = []
                        file_annotations[wav_file_name][span].extend(labels)

    # create dataset from annotations and Whisper transcriptions
    model = whisper.load_model("tiny")#"large-v2")
    dataset = dict(text=[], labels=[])
    for wav_file_name, annotations in file_annotations.items():
        wav_path = os.path.join(wav_file_dir, wav_file_name)
        audio = whisper.load_audio(wav_path)
        for ((turn_start, turn_end), labels) in sorted(annotations.items()):
            turn_start = int(turn_start * whisper.audio.SAMPLE_RATE)
            turn_end = int(turn_end * whisper.audio.SAMPLE_RATE)
            turn_audio = audio[turn_start:turn_end]
            whisper_json = whisper.transcribe(model, turn_audio)
            text = whisper_json["text"]
            dataset["text"].append(text)
            dataset["labels"].append(labels)

    dataset = datasets.Dataset.from_dict(dataset)
    dataset.save_to_disk(dataset_path)


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
    dataset_parser.add_argument("dataset_path")

    kwargs = vars(parser.parse_args())
    kwargs.pop("func")(**kwargs)
