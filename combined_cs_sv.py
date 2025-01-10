#### This code introduces the method of concatenating cs and sv. 
#### It is processed on the original dataset. 
#### It only needs to be run once to obtain a new combined dataset and save it locally.
#### This code is based on the operation of Kaggle, and some operations in the file name may be special.


# Step 1: Remove individuals with only one modality of voice recordings (Manually delete files)

# Step 2: Concatenate data based on matching IDs



import re
import os
import pandas as pd
from pydub import AudioSegment
import zipfile
import parser

args = parser.parse_arguments()

input_csv = args.data_path    # 'CASI.csv' or 'Controlli.csv'
input_folder = args.audio_folder    # folder contains audios 'CASI' or 'Controlli'
output = args.output_dir    # output folder
output_folder = os.path.join(output, "concatenated")  # path of the output concatenated audio

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Reading csv
df = pd.read_csv(input_csv)

# remove special symbols
def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9_( )./-]', '', text)

df['path'] = df['path'].apply(remove_special_characters)

# Ustore each person's file name
file_dict = {}

# Get a list of all audio files
audio_files = os.listdir(input_folder)

# Organizing file names
for file in audio_files:
    if file.endswith(".wav"):
        # Extract the prefix (i.e. remove the suffix and identification part)
        prefix = file.rsplit('_', 1)[0]

        if prefix not in file_dict:
            file_dict[prefix] = {}
        if "_cs" in file:
            file_dict[prefix]['cs'] = file
        elif "_sv" in file:
            file_dict[prefix]['sv'] = file

# Processing each person's audio file
for prefix, files in file_dict.items():
    if 'cs' in files and 'sv' in files:
        cs_filename = files['cs']
        sv_filename = files['sv']
        
        # Constructing the full file path
        cs_path = os.path.join(input_folder, cs_filename)
        sv_path = os.path.join(input_folder, sv_filename)
        
        # Read cs and sv audio files
        cs_audio = AudioSegment.from_file(cs_path)
        sv_audio = AudioSegment.from_file(sv_path)

        # Create a 1 second silence segment to separate
        silent_seperator = AudioSegment.silent(duration=1000)  # 1000 millisecond = 1 second
        
        # Example: Processing audio, here is simply splicing audio
        combined_audio = cs_audio +  silent_seperator + sv_audio

        # Build output file path
        output_filename = f"{os.path.basename(cs_filename).rsplit('_', 1)[0]}_combined.wav"
        output_file_path = os.path.join(output_folder, output_filename)
        # print(output_filename)

        # Save the processed audio file
        combined_audio.export(output_file_path, format="wav")
        
        # Update path information in CSV record
        for index, row in df.iterrows():
            if cs_filename in row['path']  or sv_filename in row['path']:
                df.at[index, 'path'] = output_filename
                

# Save the updated CSV file
df.drop_duplicates(inplace=True)
updated_csv_file = os.path.join(input_folder, "concat_file.csv")
df.to_csv(updated_csv_file, index=False)
print(f"Concatenated audio are saved in: {output_file_path}")



# ########### Compress the generated concatenated audio #######################

# Compress folders to zip files
# def zip_folder(folder_path, zip_path):
#     with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

# # Call function to compress folder
# zip_file_path = os.path.join(input_folder, "controlli_concat.zip") 
# zip_folder(output_folder, zip_file_path)