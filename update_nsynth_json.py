import os
import shutil
import glob
import json

def distribute_files(audio_dir, json_dir):

    #Find all audio files in the input directory
    audio_files = sorted(glob.glob(os.path.join(audio_dir, '**/*.wav'), recursive=True))

    print('found number of files: ' + str(len(audio_files)))

    #load JSON

    with open(os.path.join(json_dir, 'examples.json')) as f:
        json_dict = json.load(f)

    #Cicle on raw files
    for filename in audio_files:
        keyname = os.path.basename(filename)[:-4]
        subdir_name = os.path.basename(os.path.dirname(filename))
        try:
            json_dict[keyname]['subdir'] = subdir_name
        except KeyError:
            print('Key not found for: ' + keyname)


    output_json_file_path = './Datasets/nsynth-train-redist/examples-redit.json'
    with open(output_json_file_path, 'w') as file:
        json.dump(json_dict, file, indent=4)

if __name__ == "__main__":
    audio_dir = "./Datasets/nsynth-train-redist/audio"
    json_dir = "./Datasets/nsynth-train-redist/"

    distribute_files(audio_dir, json_dir)
