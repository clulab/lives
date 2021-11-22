"""json_parser.py

Parses a LabelStudio JSON export and converts it into single JSON and CSV files.

- Data exports are contained in single files for each "Project" (usually known as `annotation task`)
- We want to divide these large exports into simple, pretty printed JSON files for each "Task" (usually known as `annotated file`)
- We also want to extract data from these JSON exports and store them in a usable format (csv)

Input:
  - A single LabelStudio JSON export file from '../data/ls_export_file.json'
      The input data format is also described in `README_labeling_interface.md`

Output:
  - One pretty-printed JSON file per task for easy inspection and storage
  - Three csv table with the format described in `README_labeling_interface.md`:
    - A diarisation csv file
    - A turn annotation csv file
    - An interview annotation csv file

Notes:
        The code if not very straightforward because
        LabelStudio data JSON exports have very bad naming conventions.
        For example, the label 'id' is used for the "task", the "annotator" and the "annotation" alike

        Input files must have been exported from LabelStudio as JSON
        The JSON-MIN export format does not have enough information to be useful.

        # TODO Add skips for "bookmarks", "to discuss", etc. and add extra module for data extraction from .wav files

Damian Romero, Fall 2021, LIvES project
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from argparse import ArgumentParser
from pathlib import Path, PosixPath
from typing import Optional

from commons.lives_exceptions import AnnotationBaseError


################ ANNOTATION DETAILS ##################

def get_annotation_instance_details(annotation_instance: dict):
    """Just like `get_interview_details` but for each annotator's annotations
    """
    # This instance of annotator's annotations
    annotation_instance_id = annotation_instance["id"] 
    # This annotator's UA NETID  
    annotator_id = annotation_instance["completed_by"]["email"].removesuffix('@email.arizona.edu') 
    return annotation_instance_id, annotator_id


def get_interview_details(interview: dict,
                          get_interview_length: bool):
    """Retrieve interview details form JSON dictionary
    Output:
      wav_file_name = interview['file_upload']
        audio file name
      interview_id = interview['id']
        Annotation items are called "tasks" in LabelStudio
      task_id = interview['project']
        Annotation tasks are called "projects" in LabelStudio
    """
    if get_interview_length:
        return interview['file_upload'], interview['id'], interview['project'], interview['interview_length']
    return interview['file_upload'], interview['id'], interview['project']


def get_annotation_details(annotator_annotations_dict: dict,
                              get_interview_length: bool = False) -> tuple:
    """Returns annotation_id, variable_type, variable_name, and annotations
    """
    annotations_dict = {}
    
    variable_type = annotator_annotations_dict["type"]  # We are interested in 4 types: choices, labels, rating, textarea

    # Deal with "relation" which are created by mistake when dragging turns into each other in LabelStudio
    if variable_type == 'relation':
        annotation_id = None
        variable_name = None
        annotations = None
        print (AnnotationBaseError(variable_type, "variable_type", annotator_annotations_dict))
        return annotation_id, variable_type, variable_name, annotations

    annotation_id = annotator_annotations_dict["id"]  # Created by LabelStudio. All but `rating` ids correspond to each turns

    # Annotation IDs of "labels" define the turns, which also identify the 'choices' and 'textarea' types
    if variable_type == "labels":
        variable_name = "labels"
        annotations = annotator_annotations_dict["value"]["labels"]
        # TODO: Add unit tests to make sure only one speaker is labeled per turn and turns do not repeat
    # Populate annotation dictionary with choice annotations
    elif variable_type == "choices":
        variable_name = annotator_annotations_dict["from_name"]
        annotations = annotator_annotations_dict["value"]["choices"]
    # Populate annotation dictionary with text annotations
    elif variable_type == "textarea":
        variable_name = annotator_annotations_dict["from_name"]
        annotations = annotator_annotations_dict["value"]["text"]
    # Interview level annotations will be printed after this loop, so we need to add annottator_id
    elif variable_type == "rating":
        variable_name = annotator_annotations_dict["from_name"]
        annotations = annotator_annotations_dict["value"]["rating"]
    else:
        raise AnnotationBaseError(variable_type, "variable_type", annotator_annotations_dict)
    return annotation_id, variable_type, variable_name, annotations


def get_annotator_annotations(annotator_annotations_dict: dict,
                              annotator_id: str,
                              interview_length: Optional[float] = None) -> tuple:
    """Loop over annotator annotations and put them together inside lists of dictionaries
    """

    turn_diarisation_list = []  # Turn information (annotation_id, start, end, speaker label, etc)
    turn_annotation_list = []  # List of dictionaries for turn annotations 
    interview_annotation_list = []  # List of dictionaries for interview annotations

    for result in annotator_annotations_dict["result"]:

        # Get annotation details (annotation_id, variable_type, variable_name, annotations)
        annotation_id, variable_type, variable_name, annotations = get_annotation_details(annotator_annotations_dict = result)

        if variable_type == 'relation': # Skip relation types
            print (AnnotationBaseError(result["type"], "variable_type", result))
            continue

        # Annotation IDs of "labels" define the turns, which also identify the 'choices' and 'textarea' types
        if variable_type == "labels":
            # All `labels` annotation types contain the interview `original_length`
            if not interview_length:
                interview_length = result["original_length"]
            # TODO: Add unit tests to make sure only one speaker is labeled per turn and turns do not repeat
            turn_start = result["value"]["start"]
            turn_end = result["value"]["end"]
            turn_diarisation_list.append({"annotation_id": annotation_id,
                                          "speaker": annotations,
                                          "turn_start": turn_start,
                                          "turn_end": turn_end,
                                          "turn_length": turn_end - turn_start})
        # Populate annotation dictionary with choice annotations
        elif variable_type == "choices":
            turn_annotation_list.append({"annotation_id": annotation_id,
                                         "variable_type": variable_type,
                                         "variable_subtype": variable_name,
                                         "annotations": annotations})
        # Populate annotation dictionary with text annotations
        elif variable_type == "textarea":
            turn_annotation_list.append({"annotation_id": annotation_id,
                                         "variable_type": variable_type,
                                         "variable_subtype": variable_name,
                                         "annotations": annotations})
        # Interview level annotations will be printed after this loop, so we need to add annottator_id
        elif variable_type == "rating":
            # Interview-level annotations should be: { interview{ annotator{ global_construct, rating}}}
            interview_annotation_list.append({"annotation_id": annotation_id,
                                              "variable_type": variable_type,
                                              "variable_subtype": variable_name,
                                              "rating": annotations})
    if interview_length:
        return turn_diarisation_list, turn_annotation_list, interview_annotation_list, annotator_id, interview_length
    return turn_diarisation_list, turn_annotation_list, interview_annotation_list, annotator_id, "UNKNOWN"


################ OUTPUT FUNCTIONS ########################

def save_individual_interview(output_prefix_path: str,
                              wav_file_name: str,
                              interview_id: str,
                              interview: dict) -> None:
    """Save each interview in its own JSON file

    This is the exact same LabelStudio format but with added `interview_length` annotation
    https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks
    """

    if __name__ == '__main__':
        output_interview_json = f"{wav_file_name.removesuffix(r'.wav')}_ID{interview_id}_bckp.json"
    else:
        output_interview_json = f"{wav_file_name.removesuffix(r'.wav')}_ID{interview_id}.json"
    
    out_path = output_prefix_path.joinpath(output_interview_json)

    with open(out_path, "w") as out_json:
        json.dump(interview, out_json, ensure_ascii=False, indent=4)
        print(f"** Saved File {output_interview_json}**\n")


def make_turn_diarisation_file(turn_diarisation_dict: dict,
                               csv_write_diarisation_path: PosixPath,
                               heading_prefix: str,
                               csv_prefix: str,
                               start: bool):
    """Create csv file output for diarisation results
    TODO Add function to check which turns overlap and have a list in a column in the csv
    """
    # {'annotation_id': 'wavesurfer_2em6b8b589', 'speaker': ['coach'], 'turn_start': 1023.1140962417152, 'turn_end': 1024.739251627276, 'turn_length': 1.6251553855608}
    CSV_HEADINGS_TURN_DIARISATION = ",speaker_type,turn_start,turn_end,turn_length,is_overlapped,overlapping\n"
    if start:
        with open(csv_write_diarisation_path, 'w') as out_file:
            out_file.write(f"{heading_prefix}{CSV_HEADINGS_TURN_DIARISATION}")
            for annotator in turn_diarisation_dict:
                for annotation in turn_diarisation_dict[annotator]:
                    fields = f"{csv_prefix},{annotator},{annotation['annotation_id']},{annotation['speaker'][0]},{annotation['turn_start']},{annotation['turn_end']},{annotation['turn_length']}\n"
                    out_file.write(fields)
    else:
        with open(csv_write_diarisation_path, 'a') as out_file:
            for annotator in turn_diarisation_dict:
                for annotation in turn_diarisation_dict[annotator]:
                    fields = f"{csv_prefix},{annotator},{annotation['annotation_id']},{annotation['speaker'][0]},{annotation['turn_start']},{annotation['turn_end']},{annotation['turn_length']}\n"
                    out_file.write(fields)


def make_turn_annotation_file(turn_annotation_dict: dict,
                               csv_write_turn_annotation_path: PosixPath,
                               heading_prefix: str,
                               csv_prefix: str,
                               start: bool):
    """Create csv file output for turn annotation results
    """
    # {'annotation_id': 'wavesurfer_baj8j1v64v', 'variable_type': 'choices', 'variable_subtype': 'participant_mi_related', 'annotations': ['sustain_talk_challenges_and_barriers', 'change_talk_goal_talk_and_opportunities']}
    CSV_HEADINGS_TURN_ANNOTATION = ",variable_type,variable_subtype,annotation\n"
    if start:
        with open(csv_write_turn_annotation_path, 'w') as out_file:
            out_file.write(f"{heading_prefix}{CSV_HEADINGS_TURN_ANNOTATION}")
            for annotator in turn_annotation_dict:
                for annotation in turn_annotation_dict[annotator]:
                    for instance in annotation['annotations']:
                        instance = instance.rstrip().replace(',','/')  # TODO: Quote via CSV module instead
                        fields = f"{csv_prefix},{annotator},{annotation['annotation_id']},{annotation['variable_type']},{annotation['variable_subtype']},{instance}\n"
                        out_file.write(fields)
    else:
        with open(csv_write_turn_annotation_path, 'a') as out_file:
            for annotator in turn_annotation_dict:
                for annotation in turn_annotation_dict[annotator]:
                    for instance in annotation['annotations']:
                        instance = instance.rstrip().replace(',','/')  # TODO: Quote via CSV module instead
                        fields = f"{csv_prefix},{annotator},{annotation['annotation_id']},{annotation['variable_type']},{annotation['variable_subtype']},{instance}\n"
                        out_file.write(fields)


def make_interview_annotation_file(interview_annotation_dict: dict,
                                   csv_write_interview_annotation_path: PosixPath,
                                   heading_prefix: str,
                                   csv_prefix: str,
                                   start: bool):
    """Create csv file output for interview annotation results
    """
    # {'annotation_id': 'BRIPZjmNit', 'variable_type': 'rating', 'variable_subtype': 'interview_autonomy_support', 'rating': 5}
    CSV_HEADINGS_INTERVIEW_ANNOTATION = ",variable_type,variable_subtype,rating\n"
    if start:
        with open(csv_write_interview_annotation_path, 'w') as out_file:
            out_file.write(f"{heading_prefix}{CSV_HEADINGS_INTERVIEW_ANNOTATION}")
            for annotator in interview_annotation_dict:
                for annotation in interview_annotation_dict[annotator]:
                    fields = f"{csv_prefix},{annotator},{annotation['annotation_id']},{annotation['variable_type']},{annotation['variable_subtype']},{annotation['rating']}\n"
                    out_file.write(fields)
    else:
        with open(csv_write_interview_annotation_path, 'a') as out_file:
            for annotator in interview_annotation_dict:
                for annotation in interview_annotation_dict[annotator]:
                    fields = f"{csv_prefix},{annotator},{annotation['annotation_id']},{annotation['variable_type']},{annotation['variable_subtype']},{annotation['rating']}\n"
                    out_file.write(fields)


################ PARSE INTERVIEWS ########################

def turn_sort_overlap(in_dict: dict):
    # TODO: Implement this for the diarisation csv
    """Sort turns and check for overlaps

    output:
      turn_metadata_dict: dict
        {annotator_id: number of annotated turns}
      sorted_dict: dict of [lists sorted by `turn_start`]
        {annotator_id: [(annotation_id, turn_start, turn_end)]}
      annotation_overlaps_dict: dict
        {annotation_id:[annotation_id of overlapping turns]}
    """
    turn_metadata_dict = {}
    sorted_dict = {}
    annotation_overlaps_dict = {}
    for annotator in in_dict:
        to_sort_list = [] # list of tuples (annotation id, start, end)
        for annotation in in_dict[annotator]:
            start = in_dict[annotator][annotation]["turn_start"]
            end = in_dict[annotator][annotation]["turn_end"]
            to_sort_list.append((annotation, start, end))

        # Sort list and perform checks
        sorted_dict[annotator] = sorted(to_sort_list, key=lambda tup: tup[1])
        # for i in range(0, len(to_sort_dict[annotator])-1):
        turn_metadata_dict[annotator] = {"number_of_turns": len(sorted_dict[annotator])}
        for i in range(0, len(sorted_dict[annotator])-1):
            assert sorted_dict[annotator][i][1] <= sorted_dict[annotator][i+1][1]
            print(f"{sorted_dict[annotator][i][1]} <= {sorted_dict[annotator][i+1][1]}")

            # # Check if annotations overlap
            # annotation_overlaps = []
            # for j in range(i+1, len(to_sort_dict[annotator])-1):
    print(f"{turn_metadata_dict}")
    return turn_metadata_dict, 


def parse_individual_interview_details(ind_interview_data: dict):
    """Prepares files created with `create_individual_json_files()` to transform into predictions"""

    # Get interview-level details
    wav_file_name, interview_id, task_id, interview_length = get_interview_details(ind_interview_data, True)

    print(f"\n * Processing interview {wav_file_name},\
            LabelStudioID# {interview_id},\
            ProjectID# {task_id},\
            InterviewLength: {interview_length} *")

    # `item is a dict containing project-level details and a list of annotation instances
    for item in ind_interview_data:

        # This ind_interview_data['annotations'] is the list of annotation instances, one per annotator
        if item == 'annotations': 
            # Loop through each annotator's annotations
            for annotator_annotations_dict in ind_interview_data[item]:

                # Get the details of this annotator's annotations
                annotation_instance_id, annotator_id = get_annotation_instance_details(annotator_annotations_dict)
                print(f"\n * Working on `{annotator_id}'s` annotations  *\n")

                # We'll place the following list under the "result" key to the `predictions_list[{}]` later
                result_list = [] 

                # `annotations` are a single or a group of annotations with a unique ID
                for annotations in annotator_annotations_dict["result"]:
                    # Get annotation details (annotation_id, variable_type, variable_name, annotations)
                    annotation_id, *annotation_details_list = get_annotation_details(annotator_annotations_dict = annotations)

                    yield wav_file_name, task_id, annotator_id, annotations, annotation_id, annotation_details_list


def parse_full_ls_export_file(individual_files: Optional[bool] = None,
                              diarisation_csv: Optional[bool] = None,
                              turn_anno_csv: Optional[bool] = None,
                              interview_anno_csv: Optional[bool] = None,
                              ls_exported_file_path: Optional[PosixPath] = None,
                              output_prefix_path: Optional[PosixPath] = None,
                              csv_write_diarisation_path: Optional[PosixPath] = None,
                              csv_write_turn_annotation_path: Optional[PosixPath] = None,
                              csv_write_interview_annotation_path: Optional[PosixPath] = None,
                              verbose: bool = False,
                              test: bool = False) -> None:

    with open(ls_exported_file_path, 'r') as in_json:
        jsonData = json.load(in_json)

        start = True  # Quick hack for easily writing and appending to files
        for interview in jsonData:  # `interview`s are dictionaries

            wav_file_name, interview_id, task_id= get_interview_details(interview = interview,
                                                                         get_interview_length = False)

            # We cannot get the interview length at this level, so we'll extract it from the next one
            interview_length = None

            # Create dictionaries for dumps
            turn_diarisation_dict = {}  # Turn information (annotation_id, start, end, speaker label, etc)
            turn_annotation_dict = {}  # Dictionary of dictionaries for turn annotations 
            interview_annotation_dict = {}  # Dictionary of dictionaries for interview annotations

            for annotators_dict_list in interview['annotations']:  # Each annotator's annotations is a list of dictionaries

                annotation_instance_id, annotator_id = get_annotation_instance_details(annotators_dict_list)

                print(f"* Processing task# {task_id}, interview# {interview_id}, annotator: {annotator_id}, audio file: {wav_file_name}*")

                (turn_diarisation_list,
                 turn_annotation_list,
                 interview_annotation_list,
                 annotator_id,
                 interview_length) = get_annotator_annotations(annotator_annotations_dict = annotators_dict_list,
                                                               annotator_id = annotator_id,
                                                               interview_length = interview_length)
            

                turn_diarisation_dict[annotator_id] = turn_diarisation_list
                turn_annotation_dict[annotator_id] = turn_annotation_list
                interview_annotation_dict[annotator_id] = interview_annotation_list

            interview["interview_length"] = interview_length


            # Create headings for CSV files
            CSV_HEADINGS_PREFIX = "task_id,wav_file_name,interview_id,annotator_id,annotation_id"
            csv_common_prefix = f"{task_id},{wav_file_name},{interview_id}"


            # Create individual files from the large JSON export from LabelStudio
            if individual_files:
                save_individual_interview(output_prefix_path=output_prefix_path,
                                          wav_file_name=wav_file_name,
                                          interview_id=interview_id,
                                          interview=interview)

            # Create csv file output for diarisation
            if diarisation_csv:
                make_turn_diarisation_file(turn_diarisation_dict=turn_diarisation_dict,
                                           csv_write_diarisation_path=csv_write_diarisation_path,
                                           heading_prefix=CSV_HEADINGS_PREFIX,
                                           csv_prefix=csv_common_prefix,
                                           start=start)

            # Create csv file output for turn-level annotations
            if turn_anno_csv:
                make_turn_annotation_file(turn_annotation_dict=turn_annotation_dict,
                                          csv_write_turn_annotation_path=csv_write_turn_annotation_path,
                                          heading_prefix=CSV_HEADINGS_PREFIX,
                                          csv_prefix=csv_common_prefix,
                                          start=start)

            # Create csv file output for interview-level annotations
            if interview_anno_csv:
                make_interview_annotation_file(interview_annotation_dict=interview_annotation_dict,
                                   csv_write_interview_annotation_path=csv_write_interview_annotation_path,
                                   heading_prefix=CSV_HEADINGS_PREFIX,
                                   csv_prefix=csv_common_prefix,
                                   start=start)
            if test:
                break
            
            start = False


def run(individual_bckps:bool = False,
        diarisation_csv:bool = False,
        turn_anno_csv:bool = False,
        interview_anno_csv:bool = False,
        ls_exported_file_path: Optional[PosixPath] = None,
        output_prefix_path: Optional[PosixPath] = None,
        csv_write_diarisation_path: Optional[PosixPath] = None,
        csv_write_turn_annotation_path: Optional[PosixPath] = None,
        csv_write_interview_annotation_path: Optional[PosixPath] = None,
        verbose: bool = False,
        test: bool = False) -> None:


    # Parsing full files goes here
    parse_full_ls_export_file(individual_files = individual_bckps,
                     diarisation_csv = diarisation_csv,
                     turn_anno_csv = turn_anno_csv,
                     interview_anno_csv = interview_anno_csv,
                     ls_exported_file_path = ls_exported_file_path,
                     output_prefix_path  = output_prefix_path,
                     csv_write_diarisation_path = csv_write_diarisation_path,
                     csv_write_turn_annotation_path = csv_write_turn_annotation_path,
                     csv_write_interview_annotation_path = csv_write_interview_annotation_path,
                     verbose = False,
                     test = False)

    # TODO Separate parsing from output here

    # TODO Getting descriptive stats goes here


if __name__ == '__main__':

    curr_dir = Path.cwd()

    OUT_PREFIX = 'output'
    output_prefix_path = curr_dir.joinpath(OUT_PREFIX)

    ################ LABEL STUDIO IMPORT FILE  ############

    # Define what JSON file full export LabelStudio to use when creating individual files:
    LS_EXPORTED_FILE_DIR_PARENT = "data"
    LS_EXPORTED_FILE_NAME = "project-3-at-2021-10-26-17-00-4a12e13e.json" #input_json_file
    ls_exported_file_path = curr_dir.joinpath(LS_EXPORTED_FILE_DIR_PARENT).joinpath(LS_EXPORTED_FILE_NAME)

    ################# OUTPUT CSV FILES  ##################

    # Define the paths to write the resulting CSV files:
    CSV_WRITE_DIARISATION_FILE = f"turn_diarisation.csv"
    CSV_WRITE_TURN_ANNOTATION_FILE = f"turn_annotation.csv"
    CSV_WRITE_INTERVIEW_ANNOTATION_FILE = f"interview_annotation.csv"

    csv_write_diarisation_path = curr_dir.joinpath(output_prefix_path).joinpath(CSV_WRITE_DIARISATION_FILE)
    csv_write_turn_annotation_path = curr_dir.joinpath(output_prefix_path).joinpath(CSV_WRITE_TURN_ANNOTATION_FILE)
    csv_write_interview_annotation_path = curr_dir.joinpath(output_prefix_path).joinpath(CSV_WRITE_INTERVIEW_ANNOTATION_FILE)
    
    ################ PARSE ARGS AND RUN ##################

    parser = ArgumentParser()
    parser.add_argument("-individual", "--individualized-backup", action="store_true",
                        help='Creates individual pretty-printed interview files from LabelStudio exports for backup\
                           inside ../output/')

    parser.add_argument("-diarisation", "--diarisation-csv", action="store_true",
                        help='Creates diarisation (turn detection) counts csv table from LabelStudio exports\
                           inside ../output/.')

    parser.add_argument("-turn", "--turn-annotation-csv", action="store_true",
                        help='Creates turn annotation (turn content) counts csv table from LabelStudio exports\
                           inside ../output/.')

    parser.add_argument("-interview", "--interview-annotation-csv", action="store_true",
                        help='Creates interview annotation (MITI scores) counts csv table from LabelStudio exports\
                           inside ../output/.')

    parser.add_argument("-t", "--test", action="store_true",
                        help='Use flag for test mode')

    parser.add_argument("-v", "--verbose", action="store_true",
                        help='Use flag for verbose mode')

    args = parser.parse_args()


    run(individual_bckps = args.individualized_backup,
        diarisation_csv = args.diarisation_csv,
        turn_anno_csv = args.turn_annotation_csv,
        interview_anno_csv = args.interview_annotation_csv,
        ls_exported_file_path=ls_exported_file_path,
        output_prefix_path = output_prefix_path,
        csv_write_diarisation_path = csv_write_diarisation_path,
        csv_write_turn_annotation_path = csv_write_turn_annotation_path,
        csv_write_interview_annotation_path = csv_write_interview_annotation_path,
        verbose = args.verbose,
        test = args.test)
