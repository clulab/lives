# README - LabelStudio Importing pre-annotated data

Documentation: https://labelstud.io/guide/predictions.html

There is already an importer but does not support audio annotation: https://github.com/heartexlabs/label-studio-converter

## Workflow to import past annotations as predictions

Step 1: Create new project in LabelStudio
Step 2: No need to upload data files
Step 3: In settings, add the XML annotation scheme code for the annotation interface
<!-- Step 4: Export the full JSON data from the new project to get the project ID number
    - The first ID in the JSON file, the task_id is irrelevant
    - The `"data":{"audio":` ID is the wav_file_name one that was originally uploaded -->
Step 5: Run the `annotations_import_converter.py` to create predictions
    - Turn a LabelStudio full export JSON file into individual interviews using the `json_parser.py` format
    - The individual files you want converted need to be placed into the `../data/import_data/` directory
    - This will take in the current XML code for annotation file (see above)
    - It will read all the JSON files you put in the input directory (created using `json_parser.py`)
    - It will drop any of the annotations that do not fit the current annotation scheme
    - It will create one `predictions.json` file for each interview annotations (see the format below)
Step 6: Import the `predictions.json` file. This creates new annotation tasks (interviews to annotate).
    - Check the files. If you see gaps in the turn annotations, there may have been a labeling problem
Step 7: You can create copies of the prediction by selecting the starry icon within the task
    - On the top right there will be a button called `Create Copy`
    - You can use the new copy to start annotation again. Hit submit to have your name appear.

## General format for importing audio annotations

# The first and second IDs are irrelevant when importing
# The "predictions" key cannot be specified for each annotator
# So one prediction file has to be created for each annotators' results


- Documentation: https://labelstud.io/guide/predictions.html

```json
[{
    "id":12,
    "predictions":[
        {
            "id":11,
            "result": [
                {
                    "id": "d_TPr9pjS_",
                    "type": "rating",
                    "value": {
                        "rating": 2
                    },
                    "to_name": "audio",
                    "from_name": "interview_autonomy_support"
                },
                {
                    "id": "wavesurfer_54igiu7vin8",
                    "type": "labels",
                    "value": {
                        "end": 3.7999533885311254,
                        "start": 1.5499809874271695,
                        "labels": [
                            "coach"
                        ]
                    },
                    "to_name": "audio",
                    "from_name": "labels",
                    "original_length": 815.24
                },
                {
                    "id": "wavesurfer_54igiu7vin8",
                    "type": "choices",
                    "value": {
                        "end": 3.7999533885311254,
                        "start": 1.5499809874271695,
                        "choices": [
                            "introduce_self_and_study_name"
                        ]
                    },
                    "to_name": "audio",
                    "from_name": "coach_fidelity",
                    "original_length": 815.24
                }
            ],
            "task":12
        }
    ],
    "data":{
    "audio": "https://labelstudio-dev.data7.arizona.edu/data/upload/RE1f175cf47d8bc64eba9b21aaec028e44.wav"
    },
    "project":4
}]
```

