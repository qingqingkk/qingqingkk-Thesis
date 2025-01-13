import re
import os
import pandas as pd
from pydub import AudioSegment

import argparse

from tqdm import tqdm


"""
Script for combining CS and SV audio data.

This script processes raw datasets by concatenating audio files (`cs` and `sv`) for each individual
with matching IDs. 
The resulting concatenated audio files and updated CSV metadata are saved to the specified output directory.

Features:
1. Removes individuals with only one modality of voice recordings (manual step).
2. Concatenates `cs` and `sv` audio files with optional trimming and silent separation.
3. Optionally compresses the processed files into a zip archive for easy download.

"""



def combine_cs_sv(data_path, output_dir): 
    casi_csv = os.path.join(data_path, "CASI.csv")
    contro_csv = os.path.join(data_path, "CONTROLLI.csv")
    casi_path = os.path.join(data_path, "casi")
    contro_path = os.path.join(data_path, "controlli")
    output_folder = os.path.join(output_dir, "concatenated")  # path of the output concatenated audio

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Reading csv
    casi_df = pd.read_csv(casi_csv)
    contro_df = pd.read_csv(contro_csv)

    # remove special symbols
    def remove_special_characters(text):
        return re.sub(r'[^A-Za-z0-9_( )./-]', '', text)

    casi_df['path'] = casi_df['path'].apply(remove_special_characters)
    contro_df['path'] = contro_df['path'].apply(remove_special_characters)

    # Ustore each person's file name
    file_dict = {}

    # Get a list of all audio files
    auto_folders = [casi_path, contro_path]
    audio_files =  [os.path.join(os.path.basename(folder), file) for folder in auto_folders for file in os.listdir(folder)]


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
    print('Start concatenate audios...')
    for prefix, files in tqdm(file_dict.items()):
        if 'cs' in files and 'sv' in files:
            cs_filename = files['cs']
            sv_filename = files['sv']
            
            # Constructing the full file path
            cs_path = os.path.join(data_path, cs_filename)
            sv_path = os.path.join(data_path, sv_filename)
            
            # Read audio files
            cs_audio = AudioSegment.from_file(cs_path)
            sv_audio = AudioSegment.from_file(sv_path)

            # Trim cs to 19 seconds and sv to 18 seconds 
            # Same length as single-mode method, manually adjust according to your needs
            cs_audio = cs_audio[:19000]
            sv_audio = sv_audio[:18000]

            # Create a 1 second silence segment to separate
            silent_seperator = AudioSegment.silent(duration=1000)  # 1000 millisecond = 1 second
            
            # concatenated audio
            combined_audio = cs_audio +  silent_seperator + sv_audio

            # output
            output_filename = f"{os.path.basename(cs_filename).rsplit('_', 1)[0]}_combined.wav"
            output_file_path = os.path.join(output_folder, output_filename)

            # Save
            combined_audio.export(output_file_path, format="wav")
            
            # Update path information in CSV record
            for index, row in casi_df.iterrows():
                if cs_filename in row['path']  or sv_filename in row['path']:
                    casi_df.at[index, 'path'] = os.path.join('concatenated', output_filename)
    print(f'Sucessed! the audio file are saved in {output_folder}')

    # Save the updated CSV file
    casi_df.drop_duplicates(inplace=True)
    updated_csv_file = os.path.join(output_dir, "concat_dataset.csv")
    casi_df.to_csv(updated_csv_file, index=False)
    print(f"Concatenated audio and CSV file are saved in: {updated_csv_file}")



    # # ########### Compress the generated concatenated audio if you need download them #######################

    # # Compress folders to zip files
    # if args.save_compress:
    #     def _zip_folder(audio_path, csv_file, zip_path):
    #             print('Start compress data...')
    #             with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    #                 for root, _, files in os.walk(audio_path):
    #                     for file in tqdm(files):
    #                         zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), audio_path))
    #                     if os.path.exists(csv_file):
    #                         zipf.write(csv_file, os.path.basename(csv_file))
    #                     else:
    #                         raise FileNotFoundError(f"Cannot find {csv_file}")
    #             print(f'Compressed file are save in {zip_file_path}')
    #     # Call function to compress folder
    #     zip_file_path = os.path.join(output_folder, "CS_SV_concat.zip") 
    #     _zip_folder(output_folder, updated_csv_file, zip_file_path)

def parse_arguments():    

    parser = argparse.ArgumentParser(description="Combine cs and sv data")

    # dataset parameters
    parser.add_argument("--data_path", type=str, required=True, default="./datasets", help="Path of dataset csv")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model and results")
    # parser.add_argument("--save_compress", action="store_true", help="Compress the combined data and csv file in order to download them")

    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f'Path: {args.data_path} does not exists')
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f'Path: {args.output_dir} does not exists') 
    
    return args

if __name__ == '__main__':
    args = parse_arguments()
    data_path = args.data_path
    output_dir = args.output_dir
    combine_cs_sv(data_path, output_dir)