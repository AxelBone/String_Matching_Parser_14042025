import os

from application.ExtractHPOs import Extract_HPO_Fr_StringMaching
from application.Global_objects import HPO_TERMS,AC_OBJ
import numpy as np
import pandas as pd
import re
import csv
from nltk import ngrams,RegexpTokenizer
from nltk.tokenize.util import regexp_span_tokenize
import json

# Directory containing the files
input_directory = "C:\\Users\\baddour\\Desktop\\ACUITEE-main\\application\\for_test\\"
output_directory = "C:\\Users\\baddour\\Desktop\\ACUITEE-main\\application\\ann_clean_reports_new\\"
output_directory_empty = "C:\\Users\\baddour\\Desktop\\ACUITEE-main\\application\\clean_reports_empty\\"
output_directory_exceptions = "C:\\Users\\baddour\\Desktop\\ACUITEE-main\\application\\clean_reports_exceptions\\"
# output_directory = "/appli/ENLIGHTOR4CHU/data"
# /appli/ENLIGHTOR4CHU/ENLIGHTOR_Resources/input_data/ehop/ehop_reports
# /appli/ENLIGHTOR4CHU/data
# Iterate through each file in the directory
def file_exists_in_folder(file_name):
    file_path = os.path.join(output_directory, file_name)
    return os.path.exists(file_path)
 
for filename in os.listdir(input_directory):
    print("File Name: ",filename)
    # Construct the full path to the file
    filepath = os.path.join(input_directory, filename)

    # Check if the 'file' is actually a file and not a directory
    if os.path.isfile(filepath):
        f_name = filename[:-4]
        json_file_name = output_directory + 'ann_' + f_name +'.json'
        # Open and read the file
        with open(filepath, 'r',encoding="utf-8") as file:
            # Read the contents of the file
            file_contents = file.read()
            # Do something with the file contents
            try:
                x, y = Extract_HPO_Fr_StringMaching(file_contents)

                # print("annotations_final_compact ",filename, ":", x)
                # print("annotations_final ",filename, ":", y)
                formatted_data = []
                # formatted_result = y.split("\n")
                # print("++++++++++++")
                # print(formatted_result)
                annotations = []

                for index, row in x.iterrows():
                    # Access individual elements of the row
                    start = row['start']
                    length = row['length']
                    phrase = row['phrase']
                    hpo_id = row['HPO_ID']
                    hpo_terms = row['HPO_Terms']
                    score = row['score']
                    negated = False
                    # Do something with the row data
                    # print("Index:", index)
                    # print("Start:", start)
                    # print("Length:", length)
                    # print("Phrase:", phrase)
                    # print("HPO_ID:", hpo_id)
                    # print("HPO_Terms:", hpo_terms)
                    # print("Score:", score)
                    annotation = {
                        "start": int(start),
                        "length": int(length),
                        "sentence": phrase,
                        "negated": negated,
                        "concerned_person": 'Patient',
                        "mult_CS": False,
                        "hpoAnnotation": [
                            {
                                "hpoId": hpo_id,
                                "hpoName": phrase,
                                "parser": "SM",
                                "rating": 3,
                                "ratingInit": 3
                            }
                        ]
                    }
                    print(annotation)
                    annotations.append(annotation)
                # annotations.to_json(filename+'.json', orient='records', lines=True)
                # Specify the file path where you want to save the JSON file
                    f_name = filename[:-4]
                    json_file_path = output_directory + 'ann_' + f_name +'.json'
                if annotations != []:
                    # Open the file in write mode and save the list as JSON
                    with open(json_file_path, 'w') as json_file:
                        json.dump(annotations, json_file)
                else:
                    f_name = filename[:-4]
                    ex_file_path = output_directory_empty + f_name +'.txt'
                    with open(ex_file_path, 'w') as file:
                        file.write(file_contents)


            except:
            
                f_name = filename[:-4]
                ex_file_path = output_directory_exceptions + f_name +'.txt'
                with open(ex_file_path, 'w') as file:
                    file.write(file_contents)
                   
  