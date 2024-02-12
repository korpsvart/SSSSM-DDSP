import os
import shutil
import glob

def distribute_files(input_dir, output_dir, num_subdirectories):
    #Create subdirectories

    print("ao?")
    subdirectories = [os.path.join(output_dir, f'subdir_{i+1}') for i in range(num_subdirectories)]
    for subdir in subdirectories:
        os.makedirs(subdir, exist_ok=True)

    #Find all audio files in the input directory
    audio_files = sorted(glob.glob(os.path.join(input_dir, '*.wav')))

    print('found number of files: ' + str(len(audio_files)))

    #Calculate the number of files to be placed in each subdirectory
    files_per_subdirectory = len(audio_files) // num_subdirectories
    remaining_files = len(audio_files) % num_subdirectories #if the number of files is not divisible

    #Distribute files to subdirectories
    end_index = 0
    for i, subdir in enumerate(subdirectories):
        start_index = i * files_per_subdirectory
        end_index = start_index + files_per_subdirectory

        for audio_file in audio_files[start_index:end_index]:
            shutil.move(audio_file, subdir)

    #Add the remaining files to the last directory, if present
    if remaining_files > 0:
        for audio_file in audio_files[end_index:end_index+remaining_files]:
            shutil.move(audio_file, subdir)
        


if __name__ == "__main__":
    input_directory = "./Datasets/nsynth-train/audio"
    output_directory = "./Datasets/nsynth-train/audio"
    number_of_subdirectories = 1000  #Adjust as needed

    distribute_files(input_directory, output_directory, number_of_subdirectories)
