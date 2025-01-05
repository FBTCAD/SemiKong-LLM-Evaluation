from datasets import load_dataset
import random
import json
import os

# Load the dataset (it will use the cached version)
dataset = load_dataset("pentagoniac/SemiKong_Training_Datset")

# Display the dataset structure
print(dataset)

# Display the first entry
#print("First entry in the train split:")
#print(dataset['train'][0])

# Inspect a random entry
#random_entry = dataset['train'][random.randint(0, len(dataset['train']) - 1)]
#print("\nRandom Entry:")
#print(random_entry)

# Save the train dataset to a JSON file
output_file_raw = "semikong_train_entries_raw.json"

# Convert the dataset to a list of dictionaries
train_data = [entry for entry in dataset['train']]

# Save the list of dictionaries to a JSON file
with open(output_file_raw, 'w') as f:
    json.dump(train_data, f)

print(f"Dataset saved successfully to '{output_file_raw}'.")

# Validate if the raw JSON file exists
if not os.path.exists(output_file_raw):
    print(f"File '{output_file_raw}' does not exist. Please check the save step.")
    exit(1)

# Path to your raw JSON file with `{...}{...}` entries
input_file = output_file_raw
output_file = "semikong_train_entries_fixed.json"

# Open the raw file and fix the formatting
with open(input_file, "r") as f:
    raw_data = json.load(f)

# Save the fixed JSON file
with open(output_file, "w") as f:
    json.dump(raw_data, f, indent=4)

print(f"Fixed JSON saved to '{output_file}'")
