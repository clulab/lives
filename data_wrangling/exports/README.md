# README: Wrangling

Interface: Label studio
Version: 1.2


- There is already a data exporter library that wrangles data, but it only supports transcription annotations:https://github.com/heartexlabs/label-studio-converter
- Also, our use case is very specific. We do three things at once:
  - (1) Full audio annotation
  - (2) Diarisation via labels using AudioPlus
  - (3) Annotation of turns

## Data export

1. When exporting data, export only using the `JSON` format: https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks
2. Any other format (including `JSON-MIN`) will not contain the full time stamps for choices data.
3. You can use the `python -m json.tool file.json` module to pretty print the contents of the file.
4. Here is a commented example of what the exported files look like at the annotator level
5. See the next section for data wrangling


```json
[
    {
        "id": interview_id:int,
        "annotations": [
            {
                "id": annotation_instance_id:int,
                "completed_by": {
                    "id": 5,
                    "email": annotator_id:str,
                    "first_name": "",
                    "last_name": ""
                },
                "result": [
                    {//type: rating annotations (annotator notes)
                        "id": annotation_id:str,
                        "type": variable_type:str,
                        "value": {
                            "rating": 4
                        },
                        "to_name": "audio",
                        "from_name": "interview_empathy"
                    },
                    {//type: label annotations (turn diarisation)
                        "id": "wavesurfer_osincp31uq8", //annotation_id is defined here and used in the choice and text annotations
                        "type": variable_type:str,
                        "value": {
                            "end": turn_start:float,
                            "start": turn_end:float,
                            "labels": [
                                speaker:str
                            ]
                        },
                        "to_name": "audio",
                        "from_name": "labels",
                        "original_length": turn_length:float
                    },
                    {//type: choice annotations with one annotations
                        "id": "wavesurfer_osincp31uq8", //see the annotation_id note in the turn diarisation example above
                        "type": variable_type:str,
                        "value": {
                            "end":  turn_start:float,
                            "start":  turn_end:float,
                            "choices": [
                                annotation:str,
                            ]
                        },
                        "to_name": "audio",
                        "from_name": variable_subtype:str,
                        "original_length": turn_length:float
                    },
                    {//type: choice annotations with many annotations
                        "id": "wavesurfer_6ju9scv2bkg", //this example uses a different turn
                        "type": "choices",
                        "value": {
                            "end": 1200.813126418873,
                            "start": 1138.973510110037,
                            "choices": [
                                "collaboration_good",
                                "direction_good",
                                "empathy_good"
                            ]
                        },
                        "to_name": "audio",
                        "from_name": "coach_constructs_good",
                        "original_length": 1699.456
                    },
                    {//type: textarea annotations (annotator notes)
                        "id": "wavesurfer_6ju9scv2bkg",
                        "type": variable_type:str,
                        "value": {
                            "end": turn_end:float,
                            "text": [
                                annotation:str,
                            ],
                            "start": turn_start:float,
                        },
                        "to_name": "audio",
                        "from_name": "coach_text_note",
                        "original_length": turn_length:float
                    }
                ],
                "was_cancelled": false,
                "ground_truth": false,
                "created_at": "2021-10-04T02:14:35.302335Z",
                "updated_at": "2021-10-11T23:42:27.552149Z",
                "lead_time": 961.164,
                "prediction": {},
                "result_count": 0,
                "task": 5
            }
        ],
        "predictions": [],
        "file_upload": wav_file_name:str,
        "data": {
            "audio": "https://labelstudio-dev.data7.arizona.edu/data/upload/RE2cddbcf83b4b27e538e185a04a18af2b.wav"
        },
        "meta": {},
        "created_at": "2021-09-30T00:44:11.592265Z",
        "updated_at": "2021-10-11T23:42:27.501597Z",
        "project": task_id:str
    },
```

## Wrangling data exports

1. Run the `label_studio_to_csv.py` Python package to batch convert Label Studio JSON files into usable CSV files:

- turn_diarisation.csv: 
- turn_annotation.csv
- interview_annotation.csv

	- wav_file_name # Twilo hash + .wav
	- interview_id  # ID in LabelStudio. Called "task" in LabelStudio
	- task_id       # Annotation task ID in LabelStudio. Called "project" in LabelStudio
	- annotator_id  # Annotator UA NETID
	- annotation_id # The ID for this annotation, created by LabelStudio
	- variable_type # 4 types: choices, labels, rating, textarea
	- audio_span
	- overlap
	- Interview Global Constructs
		- interview_autonomy_support
		- interview_collaboration
		- interview_direction
		- interview_empathy
		- interview_evocation
	- speaker
		- coach
		- participant
	- coach_constructs
		- coach_constructs_bad
		- coach_constructs_good
	- coach_fidelity
	- coach_techniques
		- coach_techniques_adherence
		- coach_techniques_other
	- coach_annotator_notes
	- coach_text_notes
	- participant_gi_symptoms
	- participant_neurological_symptoms
	- participant_physical_symptoms
	- participant_psychological_symptoms
	- participant_mi_related
	- participant_lives_goals
	- participant_annotator_notes
	- participant_text_note

## Annotation interface

The labeling interface configuration code (LABELING_CONFIG) lives in a document named `lives_interface_template.xml`.
The latest version of this document lives in Google Drive here: https://drive.google.com/drive/u/1/folders/13Rs-O6nlvYNjjJeyAGN2Pvc-hK99VCoZ

Notes:
- You can add hotkeys with or without modifier keys, for example: `hotkey="KEY"` `ctrl+KEY` and `shift+KEY`
- For text areas, there are custom shortcuts that will fill out some text for you automatically, for instance:
```xml
<View>
  <TextArea name="txt-1">
    <Shortcut alias="Silence" value="SILENCE" hotkey="ctrl+1" />
  </TextArea>
</View>
```
- To modify the labeling interface, go to your project > settings > labeling interface > Code

