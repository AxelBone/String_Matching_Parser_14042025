import json
import pandas as pd
import os

# Define the path to the folder containing the JSON file
folder_path = 'C:\\Users\\baddour\\Documents\\MITO\\texts\\output\\'  # Replace with the actual folder path
json_filename = 'p3..json'  # Replace with your actual JSON file name

# Construct the full path to the JSON file
json_file_path = os.path.join(folder_path, json_filename)

# Read the JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Create a list to store the extracted data
hpo_data = []

# Iterate through the items in the JSON file
for item in data:
    # Check if hpoAnnotation is present in the item
    if 'hpoAnnotation' in item:
        for annotation in item['hpoAnnotation']:
            # Extract hpoId and hpoName
            hpo_id = annotation.get('hpoId', [])
            hpo_name = annotation.get('hpoName', [])

            # Add each hpoId and hpoName pair to the list
            for hpo, name in zip(hpo_id, hpo_name):
                hpo_data.append([hpo, name])

# Convert the list to a DataFrame
df = pd.DataFrame(hpo_data, columns=['hpoId', 'hpoName'])

# Define the output Excel file path
excel_file_path = os.path.join(folder_path, 'hpo_data_MITO_3-.xlsx')

# Save the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False)

print(f"Data has been saved to {excel_file_path}")
