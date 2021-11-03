"""prediction_creator.py

module version=1.0
python version=3.10

Code to help importing previous annotations and/or predictions of LIvES NLP into LabelStudio
  
Overview
  1. Take in a LabelStudio exported file from a LabelStudio project
  2. Divide that file into different individual files: one per "task" (interview)
  3. Convert individual files into "predicted annotations" using the selected labeling scheme
  4. Build a report for the annotation project manager

Input
  For creating individual interview files
    1. LabelStudio "full" JSON exports.
  For creating files to be imported into LabelStudio
    1. Two LABEL_CONFIG files* (old and new) to check for differences between labeling schemes
    2. Individual interview files created using the `--create-interview-files` option
*Note: LABEL_CONFIG files contain the LabelStudio XML annotation details

Output
  1. `--parse-json` option creates individual interview files from a full JSON file
  2. `--create-predictions` option creates the files that you can import to LabelStudio

Damian Romero, Fall 2021, LIvES project
"""


import dataclasses
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from argparse import ArgumentParser
from pathlib import Path, PosixPath
from typing import Optional

from commons.lives_exceptions import AnnotationCheckError
from commons.xml_parser import make_codebook, write_codebook
from exports.json_parser import (parse_individual_interview_details,
                                                  parse_full_ls_export_file)


################ LABEL_CONFIG CODEBOOKS #############


def update_elems(el_list:list, translate_dict:dict):
    """Updates a list of configuration file elements given a dictionary

    When LABEL_CONFIG files change between projects, sometimes just names or "aliases" change
    in which case we'd like to preserve previous annotations but change the old name into
    the new one.

    Input
    -----
      el_list: list of elements to check for translation keys
      translate_dict: list of elements to check for translation keys

    Output
      el_list: transformed list of elements
    -----
      translated_list
    """

    translate_keys = [key for key in translate_dict.keys()]

    for index, element in enumerate(el_list):
        if element in translate_keys:
            el_list[index] = translate_dict[element]
    return el_list


def compare_elems(el1, el2):
    """Compares two iterators. Both must be the same length"""
    for i, j in zip(el1, el2, strict=True):
        if i != j:
            return False
    return True


def concat_type_name_annotation(type_name_annotation:list):
    """Turn annotation details to single strings for comparing against unwanted codebook entries

    Input
    -----
    type_name_annotation: list of strings from `get_annotation_details()` except annotation_id
      [annotation_type, annotation_name, annotation]
    """
    return f"{type_name_annotation[0]}_{type_name_annotation[1]}_{type_name_annotation[2]}".lower()


def compare_codebooks(previous_config_file:str,
                      current_config_file:str,
                      create_compared_codebooks_file: bool = False,
                      verbose: bool = False) -> list:
    """Compares annotation codebooks for later selection

    Using the annotation extractor from ls_data_wrangler.xml_parser,
    compares (variable_type, variable_name, annotation) tuples

    Output
    -----
      no_matches_list:
        A list of tuples containing the annotations that were not matched
      If `create_compared_codebooks_file` == True
        Creates the compared codebook inside the `../output/` directory
    """

    codebook_previous = make_codebook(previous_config_file)
    codebook_current = make_codebook(current_config_file)

    no_matches_list = []
    # We want to detect which of the old annotations do not match the new ones
    for index_old, element_old in enumerate(codebook_previous):
        element_found = False
        for index_new, element_new in enumerate(codebook_current):
            # If the comparison is true, pop from new list and pass
            if compare_elems(element_old, element_new):
              element_found = True
              codebook_current.pop(index_new)
        if not element_found:
          no_matches_list.append(element_old)
    if create_compared_codebooks_file:
        write_codebook(no_matches_list, '../output/annotation_codebook_mismatches.csv')
    if verbose:
        # TODO: Put this in report log
        print("\n * The following is the list of annotations to be excluded from `predicted` annotations: *")
        print ("This can be fixed by updating `old_to_new_config_categories` in annotations_import_converter.py")
        print(json.dumps(no_matches_list, indent=2))
    return no_matches_list


def annotation_checks(old_to_new_config_categories_dict:dict,
                      unwanted_annotation_strings_list:list,
                      annotation_details:list, annotation_id:str) -> bool:
    """Check that `annotation_details` contents are not in the `unwanted_annotations` list
    
    Input
    -----
    old_to_new_config_categories_dict
      Old LABEL_CONFIG file categories (aliases) and their corresponding new ones
    unwanted_annotation_strings_list
      List of codebook non-matches from `compare_codebooks`
    annotation_details
      The three last values from `get_annotation_details()`:
      `variable_type`, variable_name`, and `annotations`
    annotation_id
      The id for corresponding to the current annotation details

    Output
    ------
      True
        When it is OK to include the annotation
      False
       If the annotation should be discarded
    """

    # Turn all annotations into lower-case strings for comparison

    annotations_strings_list = []  # List of strings to check against list of unwanted annotations
    # We need to unpack all annotations because some are in a list
    if isinstance(annotation_details[-1], list):
        # print(annotation_details[-1])
        last_item = annotation_details.pop()
        for annotation in last_item:
            temp_list = annotation_details
            temp_list.append(annotation)

            # Translate annotations from the old format to the new format if needed
            update_elems(temp_list, old_to_new_config_categories)
            # print([annotation_details, last_item])
            annotations_strings_list.append(concat_type_name_annotation(temp_list))
    else:
        annotations_strings_list.append(concat_type_name_annotation(annotation_details))

    # Check that the annotations_strings_list are not contained in the unwanted_annotations_strings_list

    for annotation_string in annotations_strings_list:
        if annotation_string in unwanted_annotation_strings_list:
            # TODO: Put this in report log
            print("*** Warning: ", AnnotationCheckError(annotation_id, annotation_string, annotation_details))
            return False
    return True


################ WRITE FILES  ############################

def create_individual_json_files(ls_exported_file:str, output_prefix:str) -> None:
    """Create individual interview files from exported JSON files from LabelStudio

    Uses `parse_full_ls_export_file` from the `exports.ls_data_wrangler.json_parser` module
    """
    parse_full_ls_export_file(individual_files=True,
                     ls_exported_file_path = ls_exported_file,
                     output_prefix_path = output_prefix,
                     test= False)


def create_predictions_file(data:list, out_file_path: PosixPath) -> None:
    """Creates the files to be imported as predictions in LabelStudio"""
    print(f"** Creating {out_file_path} **")
    with open(out_file_path, 'w') as predictions_out:
        json.dump(data, predictions_out, indent=4)


################ RUN  ####################################

def run(pre_process: bool = False,
        process: bool = False,
        output_prefix: Optional[PosixPath] = None,
        ls_exported_file_path: Optional[PosixPath] = None,
        ind_interviews_path: Optional[PosixPath] = None,
        old_codebook_path: Optional[str] = None,
        new_codebook_path: Optional[str] = None,
        old_to_new_config_categories: Optional[dict] = None,
        verbose: bool = False) -> None:
    """Creates and parses individual interview files

    If pre_process: True
      Creates individual interview files from `ls_exported_file_path` to `output_prefix`
    
    If process: True
      Prepares files created with `create_individual_json_files()` to transform into predictions
    """

    if pre_process:
        print("\n* Working on your individual interview files *\n")
        create_individual_json_files(ls_exported_file=ls_exported_file_path,
                                     output_prefix=output_prefix)

    if process:

        # Compare two LABEL_CONFIG files to get the unwanted categories used in `annotation_checks`
        unwanted_annotations = compare_codebooks(previous_config_file = old_codebook_path,
                                                 current_config_file = new_codebook_path,
                                                 verbose=verbose)

        # Turn all unwanted_annotations into lower-case strings for later comparisons
        unwanted_annotation_strings_list = []
        for annotation in unwanted_annotations:
            unwanted_annotation_strings_list.append(concat_type_name_annotation(annotation))

        print("\n* Working on the files that you will import to LabelStudio *")

        # Loop through all selected interview files in `ind_interviews_path` and create an import file for each
        for ind_interview_file in ind_interviews_path.glob('*.json'):

            # We need a list to store all of the "annotations", which are dictionaries that need to go in "result"
            result_list = [] # We'll place the list under the "result" key inside `predictions_list[{}]` later

            with open(ind_interview_file, 'r') as input_json:
                jsonData = json.load(input_json)

                # Parse the interview to get the details we need
                annotation_details_iter = parse_individual_interview_details(ind_interview_data = jsonData)

                while True:
                    try:
                        (wav_file_name,
                         task_id, annotator_id,
                         annotations,
                         annotation_id,
                         annotation_details_list) = next(annotation_details_iter)
                    except:
                        break
                    else:
                        # Perform checks for each type of annotation
                        if annotation_checks(old_to_new_config_categories,
                                             unwanted_annotation_strings_list,
                                             annotation_details_list,
                                             annotation_id):
                            result_list.append(annotations)

                # We need a dictionary for the correct output format for LabelStudio
                to_output_dict = {"id_1": 1,  # This field is irrelevant of imports
                                  "predictions": [{"id":2,   # This field is irrelevant of imports
                                                   "result":result_list # add our `result_list`
                                                  }],
                                  "data":{"audio": "https://labelstudio-dev.data7.arizona.edu/data/upload/" + wav_file_name},
                                  "project":task_id  # This field is irrelevant of imports
                                  }

                # LabelStudio format requires JSON files to start as lists
                to_output = [to_output_dict]

                if verbose:
                    print("* Your file will look like this *\n")
                    print(json.dumps(to_output, indent=2))

                # Create files to be imported as predictions in LabelStudio
                predictions_file =  f"to_import_{annotator_id}_{wav_file_name.removesuffix('.wav')}.json"
                out_file_path = output_prefix.joinpath(predictions_file)
                create_predictions_file(data = to_output, out_file_path = out_file_path)


if __name__ == '__main__':

    curr_dir = Path.cwd()
    output_prefix = Path.cwd().joinpath('output')

    ################ LABEL STUDIO IMPORT FILE  ############

    # Define what JSON file full export LabelStudio to use when creating individual files:
    LS_EXPORTED_FILE_DIR_PARENT = "data"
    LS_EXPORTED_FILE_NAME = "Annotation_Pilot_(Spanish)_completed_bckp.json"

    # The path to the JSON file (as input) is usually '../data/'
    ls_exported_file_path = curr_dir.joinpath(LS_EXPORTED_FILE_DIR_PARENT).joinpath(LS_EXPORTED_FILE_NAME)

    ################# SELECTED INDIVIDUAL JSON FILES  ####

    # Define directory to read JSON single interview files from (created by the `--parse-json` option)
    IND_INTERVIEWS_DIR_PARENT = "data"
    IND_INTERVIEWS_DIR = "import_data"

    # The path to the individual interviews (as input) is usually '../data/import_data'
    ind_interviews_path = curr_dir.joinpath(IND_INTERVIEWS_DIR_PARENT).joinpath(IND_INTERVIEWS_DIR)

    ################ LABELING CONFIGURATION FILES  #######

    # Define what LABEL_CONFIG files to use. Change to match your LabelStudio projects
    LABEL_CONFIG_DIR_GRAND_PARENT = "commons"
    LABEL_CONFIG_DIR_PARENT = "labeling_config_files"

    # The path to the individual interviews (as input) is usually '../../commons/data/labeling_config_files/'
    label_config_dir_path = curr_dir.parent.joinpath(LABEL_CONFIG_DIR_GRAND_PARENT).joinpath(LABEL_CONFIG_DIR_PARENT)

    OLD_LABEL_CONFIG_FILE = "lives_interface_template_pilot.xml"
    NEW_LABEL_CONFIG_FILE = "labeling_config_IRR.xml"

    old_label_config_path = label_config_dir_path.joinpath(OLD_LABEL_CONFIG_FILE)
    new_label_config_path = label_config_dir_path.joinpath(NEW_LABEL_CONFIG_FILE)


    ################ CONFIG FILE CATEGORIES ###############

    # Define old config file categories (aliases) and their corresponding new ones:
    old_to_new_config_categories = {"rfi closed" : "rfi_narrow",
                                    "rfi_closed" : "rfi_narrow",
                                    "rfi_open" : "rfi_wide",
                                    "calories": "non_lives_goals",
                                    "chemo_brain": "chemo_brain_memory_issues"}

    ################ PARSE ARGS ##########################

    parser = ArgumentParser()
    parser.add_argument("-parse", "--parse-json", action="store_true",
                        help='Creates individual interview files from LabelStudio exports inside ../output/\
                          If you want to use a different directory, import `run()` from this module')

    parser.add_argument("-create", "--create-predictions", action="store_true",
                        help='Creates files to be imported as predictions to LabelStudio inside ../output/\
                          If you want to use a different directory, import `run()` from this module')

    parser.add_argument("-v", "--verbose", action="store_true",
                        help='Use flag for verbose mode')

    args = parser.parse_args()

    run(pre_process=args.parse_json,
        output_prefix = output_prefix,
        process=args.create_predictions,
        ls_exported_file_path=ls_exported_file_path,
        ind_interviews_path=ind_interviews_path,
        old_codebook_path=old_label_config_path,
        new_codebook_path=new_label_config_path,
        old_to_new_config_categories=old_to_new_config_categories,
        verbose=args.verbose)
