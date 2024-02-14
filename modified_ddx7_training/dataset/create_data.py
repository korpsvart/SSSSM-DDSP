import librosa
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import hydra
from pathlib import Path
from functools import partial
from shutil import copyfile
import torch
import numpy as np
import h5py
from ddx7 import spectral_ops
import operator
import functools
import json
import os
from ddx7.core import _DB_RANGE

'''
URMP Data processor class adapted from HTP paper.
https://github.com/mosheman5/timbre_painting

'''

class ProcessData():
    def __init__(self, silence_thresh_dB=40, sr=16000, device='cpu:0', seq_len=3,
                hop_size=64, max_len=4, num_mfccs=20,
                overlap = 0.0,
                debug = True,
                contiguous = True,
                contiguous_clip_noise = False):
        super().__init__()
        self.silence_thresh_dB = silence_thresh_dB
        self.sr = sr
        self.num_mfccs = num_mfccs
        self.device = torch.device(device)
        self.max_len = max_len
        self.hop_size = hop_size
        self.feat_size = self.max_len*self.sr //self.hop_size
        self.audio_size = self.max_len*self.sr
        self.overlap = overlap
        self.debug = debug
        self.contiguous = contiguous
        self.contiguous_clip_noise = contiguous_clip_noise



    def compute_mfcc(self, y):
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.num_mfccs)
        print(mfccs.shape)
        return mfccs
        
    def detect_fundamental(self, y):
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=self.sr, fmin=75, fmax=1600)
        # get indexes of the maximum value in each time slice
        max_indexes = np.argmax(magnitudes, axis=0)
        # get the pitches of the max indexes per time slice
        pitches = pitches[max_indexes, range(magnitudes.shape[1])]
        pitch = np.average(pitches)
        return pitch

    
    def extract_f0(self, audio):

        return self.detect_fundamental(audio)


    def save_data(self, audio, f0, mfcc, mfcc_length, h5f, counter):
        h5f.create_dataset(f'{counter}_audio', data=audio)
        h5f.create_dataset(f'{counter}_f0', data=f0)
        h5f.create_dataset(f'{counter}_mfcc', data=mfcc)
        h5f.create_dataset(f'{counter}_mfcc_length', data=mfcc_length) 
        return counter + 1

    def init_h5(self, data_dir):
        return h5py.File(data_dir / f'{self.sr}.h5', 'w')

    def close_h5(self, h5f):
        h5f.close()

    '''
    Main audio processing function
    '''
    def run_on_files(self, data_dir, input_dir, output_dir):
        audio_files = list((input_dir/data_dir).glob('*.wav'))
        output_dir = output_dir / data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open container
        h5f = self.init_h5(output_dir)
        counter = 0

        #num_files = len(audio_files)
        max_audio_length = 0
        max_file = None

        for audio_file in tqdm(audio_files):
            if(self.debug): print("Processing: {}".format(audio_file))

            #compute the length
            
            data, sr = librosa.load(audio_file.as_posix(), sr=self.sr)
            if data.size > max_audio_length:
                max_audio_length = data.size
                max_file = audio_file

        #Also find the mfcc length for the longest sequence
        data, sr = librosa.load(max_file.as_posix(), sr=self.sr)
        mfcc = self.compute_mfcc(data)
        max_mfcc_length = mfcc.shape[1]

        
                
            
        for audio_file in tqdm(audio_files):

            #load and pad
            data, sr = librosa.load(audio_file.as_posix(), sr=self.sr)
            audio = librosa.util.normalize(data) # Peak-normalize audio
            mfcc = self.compute_mfcc(audio)
            mfcc_length = mfcc.shape[1] #take non-padded mfcc length
            audio = librosa.util.fix_length(audio, size=max_audio_length) #pad audio signal
            mfcc = librosa.util.fix_length(mfcc, size=max_mfcc_length, axis=-1) #pad mfcc signal
                
            f0 = self.extract_f0(audio)
            
            counter = self.save_data(audio, f0, mfcc, mfcc_length, h5f, counter)

        # Finished storing f0 and loudness
        self.close_h5(h5f)


    def run_on_dirs(self, input_dir: Path, output_dir: Path):
        #print("Starting with crepe confidence: {}".format(self.crepe_params.confidence_threshold))
        folders = [x for x in input_dir.glob('./*') if x.is_dir()]
        for folder in tqdm(folders):
            self.run_on_files(folder.name, input_dir, output_dir)


def create_mono_urmp(instrument_key, audio_files, target_dir, instruments_dict):
    target_dir = target_dir / instruments_dict[instrument_key]
    if not target_dir.exists():
        target_dir.mkdir()
    cur_audio_files = [audio_file for audio_file in audio_files if f'_{instrument_key}_' in audio_file.name]
    [copyfile(audio_file, target_dir / audio_file.name) for audio_file in cur_audio_files]

def create_mono_testset(audio_files, target_dir, instrument):

    target_dir = target_dir / instrument
    if not target_dir.exists():
        target_dir.mkdir()
    cur_audio_files = [audio_file for audio_file in audio_files]
    [copyfile(audio_file, target_dir / audio_file.name) for audio_file in cur_audio_files]


def make_testset(args):
    CWD = Path(hydra.utils.get_original_cwd()) # Get current directory
    os.chdir(CWD)

    if args.testset is not None:

        if(args.skip_copy is False):

            # Create directories if needed
            dirs = args.testset.input_dir.split("/")
            target_dir = CWD
            for d in dirs:
                target_dir = target_dir / d
                #print(target_dir)
                target_dir.mkdir(exist_ok=True)

            testset_path = CWD / args.testset.source_folder
            print("[INFO] Testset source path: {}".format(testset_path))
            print("[INFO] will source files from directories: {}".format(args.testset.instruments))

            for instrument in args.testset.instruments:
                #print(instrument)
                # Find relevant audio files.
                test_audio_path = testset_path / instrument
                test_audio_files = list(test_audio_path.glob(f'./*.wav'))
                #print(test_audio_files)

                create_mono_testset(audio_files=test_audio_files,
                                    target_dir=target_dir,
                                    instrument=instrument)

    if(args.skip_process is False):

        # Create output dirs if needed
        dirs = args.testset.output_dir.split("/")
        target_dir = CWD
        for d in dirs:
            target_dir = target_dir / d
            #print(target_dir)
            target_dir.mkdir(exist_ok=True)

        # Process Test Set

        data_processor = hydra.utils.instantiate(args.data_processor)

        #Override original crepe confidence to process all the testset file.
        data_processor.set_confidence(0.0)
        data_processor.contiguous = args.testset.contiguous
        data_processor.contiguous_clip_noise = args.testset.clip_noise #Clip frequencies tracked due to noise.

        data_processor.run_on_dirs(CWD / args.testset.input_dir,
                                CWD / args.testset.output_dir)
    return

def make_urmp(args):
 # Phase 0 - copy all urmp wavs to corresponding folders
    CWD = Path(hydra.utils.get_original_cwd()) # Get current directory
    os.chdir(CWD)

    if args.urmp is not None:
        urmp_path = CWD / args.urmp.source_folder

        if(args.skip_copy is False):

            # Create directories if needed
            dirs = args.urmp.input_dir.split("/")
            target_dir = CWD
            for d in dirs:
                target_dir = target_dir / d
                #print(target_dir)
                target_dir.mkdir(exist_ok=True)

            # Find relevant audio files.
            urmp_audio_files = list(urmp_path.glob(f'./*/{args.urmp.mono_regex}*.wav'))

            print("[INFO] URMP Path: {}".format(urmp_path))

            print("[INFO] Number of files: {}".format(len(urmp_audio_files)))

            print(args.urmp.instruments.keys())
            # Partial function with instruments pre-configured for processing.
            create_mono_urmp_partial = partial(create_mono_urmp,
                                            audio_files=urmp_audio_files,
                                            target_dir=target_dir,
                                            instruments_dict=args.urmp.instruments)

            # Spawn threads to copy files.
            thread_map(create_mono_urmp_partial, list(args.urmp.instruments.keys()))

    # Process Train Set
    if(args.skip_process is False):

        # Create output dirs if needed
        dirs = args.urmp.output_dir.split("/")
        target_dir = CWD
        for d in dirs:
            target_dir = target_dir / d
            #print(target_dir)
            target_dir.mkdir(exist_ok=True)

        data_processor = hydra.utils.instantiate(args.data_processor)

        data_processor.run_on_dirs(CWD / args.urmp.input_dir,
                                CWD / args.urmp.output_dir)
    return

def make_single_note_dataset(dir_path, target_dir):

    dir_path = Path(dir_path)
    target_dir = Path(target_dir)

    print("[INFO] URMP Path: {}".format(dir_path))

    # Find relevant audio files.
    audio_file_regex = '*' #a number probably
    urmp_audio_files = list(dir_path.glob(f'./violin_laia_improvement_recordings/neumann/{audio_file_regex}.wav'))

    print("[INFO] Number of files: {}".format(len(urmp_audio_files)))

    #os.mkdir(target_dir)
    target_dir.mkdir(exist_ok=True)

    dataprocessor = ProcessData()
    

    dataprocessor.run_on_files('violin_laia_improvement_recordings/neumann/', dir_path, target_dir)



def main():

    #if(args.process_testset is True): make_testset(args)

    make_single_note_dataset('./data/good-sounds/', './data/train/')

if __name__ == "__main__":
    main()