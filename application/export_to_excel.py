import json
import pandas as pd

# Read JSON data from a file
input_file = 'C:\\Users\\baddour\\Documents\\MITO\\texts\\output\\p3..json'  # Replace with the path to your JSON file
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create an empty list to store the extracted data
extracted_data = []

# Dictionary to track how many times each "hpoId" has appeared (for iteration count)
hpo_id_counter = {}

# Set to track the hpoIds that have been added to avoid duplicates
written_hpo_ids = set()

# Loop through the data and extract "sentence", "hpoId", "hpoName" and track iteration
for entry in data:
    sentence = entry['sentence']
    for annotation in entry['hpoAnnotation']:
        hpo_ids = annotation['hpoId']
        hpo_name = annotation['hpoName']

        # Loop through each hpoId and track iteration
        for hpo_id in hpo_ids:
            # If hpoId has been seen before, just update the iteration count
            if hpo_id in written_hpo_ids:
                hpo_id_counter[hpo_id] += 1
                # Update iteration in extracted data at the position of this hpoId
                for i, row in enumerate(extracted_data):
                    if row['hpoId'] == hpo_id:
                        extracted_data[i]['iteration'] = hpo_id_counter[hpo_id]
            else:
                # Otherwise, it's a new hpoId, so start with iteration count of 1
                written_hpo_ids.add(hpo_id)
                hpo_id_counter[hpo_id] = 1
                # Insert the new row at the end of the extracted data
                extracted_data.append({
                    'sentence': sentence,
                    'hpoId': hpo_id,
                    'hpoName': hpo_name,
                    'iteration': hpo_id_counter[hpo_id]  # Iteration count for this hpoId
                })

# Convert the extracted data into a pandas DataFrame
df = pd.DataFrame(extracted_data)

# Save the DataFrame to an Excel file
excel_filename = 'hpo_annotations_with_iteration.xlsx'
df.to_excel(excel_filename, index=False)

print(f"Data has been saved to {excel_filename}")
