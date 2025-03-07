import os
import subprocess

# Define the directory containing your .ipynb files
directory = './'

# Get a list of all .ipynb files in the directory
ipynb_files = [f for f in os.listdir(directory) if f.endswith('.ipynb')]

# Convert each .ipynb file to .py
for file in ipynb_files:
    file_path = os.path.join(directory, file)
    command = f"jupyter nbconvert --to python {file_path}"
    subprocess.run(command, shell=True)
