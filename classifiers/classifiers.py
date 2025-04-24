import argparse
import collections
import json
import os.path
import pprint
import random
import re
import numpy as np
import datasets
import evaluate
import transformers
import whisper


def construct_dataset(dataset_path, wav_file_dir, project_json_paths):

    # gather LabelStudio annotations, grouping by (start, end) in the audio
    file_annotations = collections.defaultdict(dict)
    for project_json_path in project_json_paths:
        with open(project_json_path) as json_file:
            for file_json in json.load(json_file):
                wav_file_name = file_json["file_upload"]

                # fix some inconsistent name problems
                re_only_pattern = re.compile(r".*(RE\w+?)[._].*")
                suffix = '.wav' if '.wav' in wav_file_name else '.mp3'
                wav_file_name = re_only_pattern.sub(rf"\1{suffix}", wav_file_name)

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
                                 "participant_neurological_symptoms" |\
                                 "participant_gi_symptoms" |\
                                 "participant_annotator_notes" |\
                                 "coach_choices" |\
                                 "participant_choices":
                                labels = value_json["choices"]
                            case "coach_text_note" | "participant_text_note" |\
                                 "global_scores":
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
    model = whisper.load_model("tiny")#"large-v3")
    print("whisper loaded")
    dataset = dict(text=[], labels=[])
    for wav_file_name, annotations in file_annotations.items():
        print(f"processing {wav_file_name}")
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


def train(dataset_path, speaker_label, target_label,
          n_trials, n_epochs, balanced):
    pretrained_model_name = "answerdotai/ModernBERT-base"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name)

    dataset = datasets.Dataset.load_from_disk(dataset_path).map(
        lambda examples: {"label": int(target_label in examples["labels"])}
    ).filter(
        lambda example: speaker_label in example["labels"]
    ).remove_columns("labels").class_encode_column("label").map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, max_length=256),
        batched=True
    ).train_test_split(seed=42, test_size=1000, stratify_by_column="label")
    print(dataset)

    if balanced:
        # find indices for each label
        label_indices = collections.defaultdict(list)
        for i, label in enumerate(dataset['train']['label']):
            label_indices[label].append(i)
        # trim extra examples from more frequent labels
        n_indices = min(len(indices) for indices in label_indices.values())
        for label, indices in label_indices.items():
            label_indices[label] = indices[:n_indices]
        # trim dataset to match
        dataset['train'] = dataset['train'].select(sorted(
            i for indices in label_indices.values() for i in indices))

    label_counts = {d: collections.Counter(e['label'] for e in dataset[d])
                    for d in ['train', 'test']}
    pprint.pprint(label_counts, width=1)
    if any(len(set(label_counts[d])) == 1 for d in ['train', 'test']):
        raise ValueError(f"Only 1 label for {speaker_label!r} {target_label!r}")

    metrics = evaluate.combine(["f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return metrics.compute(
            predictions=np.argmax(logits, axis=1),
            references=labels,
            pos_label=1,
            average='binary')

    name = f"{pretrained_model_name}_{speaker_label}_{target_label}"
    if balanced:
        name += "_balanced"
    args = transformers.TrainingArguments(
        # learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # gradient_accumulation_steps=4,
        num_train_epochs=n_epochs,
        load_best_model_at_end=True,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_f1",
        report_to=["wandb"],
        output_dir=name)
    trainer = transformers.Trainer(
        args=args,
        model_init=lambda: transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=2,
            label2id={f"not-{target_label}": 0, target_label: 1},
            id2label={0: f"not-{target_label}", 1: target_label},
            ignore_mismatched_sizes=True),
        processing_class=tokenizer,
        data_collator=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"])

    hp_space = {
        "project": "lives",
        "method": "random",
        "name": name,
        "metric": {
            "name": "eval_f1",
            "goal": "maximize",
        },
        "parameters": {
            "learning_rate": {
                "distribution": "uniform",
                "min": 1e-5,
                "max": 1e-4},
            "seed": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 4},
        }
    }

    trainer.hyperparameter_search(
        direction="maximize",
        backend="wandb",
        n_trials=n_trials,
        hp_space=lambda trial: hp_space)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    train_parser.add_argument("dataset_path")
    train_parser.add_argument("--speaker", dest='speaker_label', required=True)
    train_parser.add_argument("--target", dest='target_label', required=True)
    train_parser.add_argument("--n-trials", type=int, default=1)
    train_parser.add_argument("--n-epochs", type=int, default=3)
    train_parser.add_argument("--balanced", action='store_true')

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.set_defaults(func=construct_dataset)
    dataset_parser.add_argument("dataset_path")
    dataset_parser.add_argument("wav_file_dir")
    dataset_parser.add_argument("project_json_paths",
                                nargs="+", metavar="project_json_path")

    kwargs = vars(parser.parse_args())
    kwargs.pop("func")(**kwargs)
