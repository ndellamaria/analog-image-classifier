import os
import pandas as pd

# Set the path to the root directory containing the class folders
root_dir = "train"

# Initialize a list to store image paths and labels
data = []

# Loop through each folder in the root directory
for label in os.listdir(root_dir):
    # Full path to the folder
    label_dir = os.path.join(root_dir, label)
    
    # Check if it's a directory (in case there are other files)
    if os.path.isdir(label_dir):
        # Loop through each image file in the class folder
        for image_file in os.listdir(label_dir):
            # Add the image path and label to the data list
            data.append([image_file, label])

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=["image_file", "label"])

# Save to a labels.txt file
df.to_csv("labels_1.txt", index=False, header=False)
