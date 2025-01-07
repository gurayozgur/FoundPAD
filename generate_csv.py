import csv
import os
import random

# Define the dataset directory and output CSV file
dataset_dir = "/home/ozgur/Datasets/SynthASpoof"  # Replace with your dataset directory
output_csv = "synthaspoof.csv"

# List to store image paths and labels
data = []

# Walk through the dataset directory
for label in ["bonafide", "attack"]:
    label_dir = os.path.join(dataset_dir, label)
    if os.path.exists(label_dir):  # Ensure the directory exists
        for root, _, files in os.walk(
            label_dir
        ):  # Recursively walk through directories
            for file_name in files:
                if file_name.endswith(
                    (".png", ".jpg", ".jpeg")
                ):  # Filter for image files
                    image_path = os.path.join(root, file_name)
                    data.append([image_path, label])
random.shuffle(data)
# Write to CSV
with open(output_csv, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["image_path", "label"])  # Write header
    writer.writerows(data)  # Write data

print(f"CSV file created: {output_csv}")
