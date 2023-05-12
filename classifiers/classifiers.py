import argparse
import collections
import json
import os.path
import pprint
import re
import datasets
import whisper


# ugly hack for ssl.SSLCertVerificationError
# https://stackoverflow.com/a/28052583/384641
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def construct_dataset(dataset_path, wav_file_dir, project_json_paths):

    # gather LabelStudio annotations, grouping by (start, end) in the audio
    file_annotations = collections.defaultdict(dict)
    for project_json_path in project_json_paths:
        with open(project_json_path) as json_file:
            for file_json in json.load(json_file):
                wav_file_name = file_json["file_upload"]

                # fix some inconsistent name problems
                re_only_pattern = re.compile(r".*(RE\w+?)[._].*")
                wav_file_name = re_only_pattern.sub(r"\1.wav", wav_file_name)

                # skip missing files
                wav_file_path = os.path.join(wav_file_dir, wav_file_name)
                if not os.path.exists(wav_file_path):
                    print(f"WARNING: skipping missing file {wav_file_path}")
                    continue

                # each annotation is a LabelStudio "result"
                # until we have adjudication, use only the first annotator's work
                annotations_json = file_json["annotations"][0]

                for result_json in annotations_json["result"]:
                    value_json = result_json.get("value")

                    # use all annotations with spans
                    if value_json is not None and "start" in value_json:
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
                                 "participant_physical_symptoms"|\
                                 "participant_psychological_symptoms" |\
                                 "participant_annotator_notes":
                                labels = value_json["choices"]
                            case "coach_text_note" | "participant_text_note":
                                pass
                            case _:
                                raise NotImplementedError(str(result_json))

                        # save span and labels
                        span = turn_start, turn_end
                        if span not in file_annotations[wav_file_name]:
                            file_annotations[wav_file_name][span] = []
                        file_annotations[wav_file_name][span].extend(labels)

    pprint.pprint(collections.Counter(
        label
        for annotations in file_annotations.values()
        for labels in annotations.values()
        for label in labels))

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
    dataset_parser.add_argument("dataset_path")
    dataset_parser.add_argument("wav_file_dir")
    dataset_parser.add_argument("project_json_paths",
                                nargs="+", metavar="project_json_path")

    kwargs = vars(parser.parse_args())
    kwargs.pop("func")(**kwargs)
