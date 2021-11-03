"""Basic script for extracting LabeStudio codes

- We need a codebook both for our documentation and to extract data from LabelStudio exports
- The codes we need for the codebook are user-defined in an xml file created by Damian Romero
- LabelStudio uses the xml file for its `Labeling interface` settings
- These settings may change during the course of this project. 
- Keeping track of the changes in the xml file codes is inefficient and prone to error
- This script will help to avoid doing manual work every time we change the Labeling interface

Input:
  directory = './labeling_config_files/'
  LabelStudio labeling interface xml code, usually in a file `lives_interface_template.xml`
Output:
  directory = './'
  UTF8 csv table with the following headings:
    variable_type, variable_name, choices, 

The codebook is usually used as documentation or as input for the `ls_data_wrangler` module
for extracting information from the LabelStudio data exports (JSON)

Damian Romero, Fall 2021, LIvES project
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

from argparse import ArgumentParser
from pathlib import Path, PosixPath
from typing import Optional


def write_codebook(codebook: list,
                   out_file_name: str = './annotation_codebook.csv') -> None: 

    CODEBOOK_HEADING = "variable_type,variable_name,annotation"
    with open(out_file_name, 'w') as out_file:
        out_file.write(f'{CODEBOOK_HEADING}\n')
        for elem in codebook:
            # print(elem)
            out_file.write(f'{elem[0]},{elem[1]},{elem[2]}\n')


def make_codebook(xml_document: str = './labeling_config_files/lives_interface_template_pilot.xml'):
    """Creates a list of tuples from a LABEL_CONFIG document

    Output: codebook_list
      format [(variable_type, variable_name, annotation)]
    """
    tree = ET.parse(xml_document)
    root = tree.getroot()

    # modify to add the types (hypernyms) of the `control tags`
    # https://labelstud.io/tags/
    variable_types = ['Choices', 'Labels', 'Rating', 'TextArea']
    variable_types_annotations = {'Rating': 'maxRating', 'TextArea': 'placeholder'}
    variable_types_elements = ([i for i in root.iter(type)] for type in variable_types)

    codebook_list = []
    for i in variable_types_elements:
        for j in i:
            if len(j) > 1:
                for k in j:
                    try:
                        k.attrib["alias"]
                    except Exception as e:
                        print(f'** Note: The following element will not be included: {j.tag}, {j.attrib["name"]}, {k.tag} ***')
                        pass
                    else:
                        codebook_list.append((j.tag,j.attrib["name"],k.attrib["alias"]))
            else:
                codebook_list.append((j.tag,j.attrib["name"],j.attrib[variable_types_annotations[j.tag]]))
    return codebook_list


def run(input_path: Optional[PosixPath] = None,
        output_path: Optional[PosixPath] = None,
        print_codebook: bool = False):

    # Create list of tuples containing the codebook
    codebook_list = make_codebook(xml_document = input_path)

    if print_codebook:
        print(f"*** Printing {input_path} codebook to {output_path} ***")
        write_codebook(codebook = codebook_list, out_file_name = output_path)


if __name__ == '__main__':

    curr_dir = Path.cwd()

    # Output file name and path
    OUT_PREFIX = 'output'
    DEFAULT_OUT_FILENAME = 'annotation_codebook.csv'
    out_path = curr_dir.joinpath(OUT_PREFIX).joinpath(DEFAULT_OUT_FILENAME)

    # Input file name and path
    LABELING_CONFIG_DIR = 'labeling_config_files'
    LABELING_CONFIG_FILE = 'labeling_config_IRR.xml'
    in_path = curr_dir.joinpath(LABELING_CONFIG_DIR).joinpath(LABELING_CONFIG_FILE)

    parser = ArgumentParser()
    parser.add_argument('--create', nargs='?', const=True, default=False, help="Optional argument for creating\
                         `annotation_codebook.csv`. If not specified, performs a dry run.")
    args = parser.parse_args()

    run(input_path = in_path, output_path = out_path, print_codebook = args.create)
