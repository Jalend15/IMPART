import os

# Define the directory to scan
directory_path = '2019/'

# Specify the file extension of files to delete
necessary_extension = '.mp4'

# Walk through the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        # Check if the file has the unnecessary extension
        if file.endswith(necessary_extension):
            # Construct the full file path
            print(file)
        else:
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Print the file path (you can remove this line in the actual usage)
            print("Deleting:", file_path)
            # Delete the file
            os.remove(file_path)
            print("Deleted")            
