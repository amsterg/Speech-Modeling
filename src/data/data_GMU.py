import sys
import os
#nopep8
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from yaml import safe_load
import pandas as pd
from collections import Counter
from scipy.ndimage import binary_dilation
import pathlib
from colorama import Fore
from tqdm import tqdm
import warnings
import h5py
from data_utils import preprocess, mel_spectogram, structure, write_hdf5
from random import sample,shuffle
import numpy as np

warnings.filterwarnings("ignore")

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

RAW_DATA_DIR = config['RAW_DATA_DIR']
PROC_DATA_DIR = config['PROC_DATA_DIR']
INTERIM_DATA_DIR = config['INTERIM_DATA_DIR']
MODEL_SAVE_DIR = config['MODEL_SAVE_DIR']
GMU_DATA_PACKED = config['GMU_DATA_PACKED']
GMU_DATA_INFO = config['GMU_DATA_INFO']
GMU_DATA = config['GMU_DATA']
GMU_ACCENT_COUNT = config['GMU_ACCENT_COUNT']
AUDIO_WRITE_FORMAT = config['AUDIO_WRITE_FORMAT']
AUDIO_READ_FORMAT = config['AUDIO_READ_FORMAT']

GMU_INT_DIR = GMU_DATA.replace(RAW_DATA_DIR, INTERIM_DATA_DIR)
GMU_PROC_OUT_FILE_T = os.path.join(
    PROC_DATA_DIR, 'gmu_{}_train.hdf5'.format(GMU_ACCENT_COUNT))
GMU_PROC_OUT_FILE_VAL = os.path.join(
    PROC_DATA_DIR, 'gmu_{}_val.hdf5'.format(GMU_ACCENT_COUNT))

dirs_ = set([globals()[d] for d in globals() if d.__contains__('DIR')])


def extract_GMU():
    """
    Summary:
    
    Args:
    
    Returns:
    
    """
    structure(dirs_)

    if not pathlib.Path.exists(pathlib.Path(GMU_DATA_PACKED)):
        #        download
        pass
    if not pathlib.Path.exists(pathlib.Path(GMU_DATA_PACKED.split('.')[0])):
        #       unzip
        pass


def preprocess_GMU():
    """
    Summary:
    
    Args:
    
    Returns:
    
    """
    speakers_info = pd.read_csv(GMU_DATA_INFO)
    categories = Counter(
        speakers_info['native_language']).most_common(GMU_ACCENT_COUNT)
    categories = [c[0] for c in categories]
    speakers_info = speakers_info[speakers_info['native_language'].isin(
        categories)]
    speakers_info = speakers_info[['filename', 'native_language']]
    speakers_info['name'] = speakers_info['filename']
    speakers_info['filename'] = speakers_info['filename'].apply(
        lambda fname: os.path.join(GMU_DATA, fname + AUDIO_READ_FORMAT))

    count = 888
    
    shuffle_ixs = list(range(len(speakers_info['filename'].tolist())))
    shuffle(shuffle_ixs)
    fnames = np.array(speakers_info['filename'].tolist())[shuffle_ixs][count:].tolist()
    langs = np.array(speakers_info['native_language'].tolist())[shuffle_ixs][count:].tolist()
    names = np.array(speakers_info['name'].tolist())[shuffle_ixs][count:].tolist()
    train_names = sample(names, int(0.8 * len(names)))
    val_names = set(names) - set(train_names)

    mels_train = {lang: [] for lang in langs}
    mels_val = {lang: [] for lang in langs}

    for name, fname, lang in tqdm(zip(names, fnames, langs),
                                  total=len(langs),
                                  bar_format="{l_bar}%s{bar}%s{r_bar}" %
                                  (Fore.GREEN, Fore.RESET)):
        try:
            aud = preprocess(fname)
        except AssertionError as e:
            print("Couldn't process ", len(aud), fname)

        # file_out_ = fname.split('.')[0].replace(
        #     RAW_DATA_DIR, INTERIM_DATA_DIR) + '_' + AUDIO_WRITE_FORMAT

        # soundfile.write(file_out_, aud, SAMPLING_RATE)

        mel = mel_spectogram(aud)
        if mel.shape[1] <= config['SLIDING_WIN_SIZE']:
            print("Couldn't process ", mel.shape, fname)
            continue
        if name in val_names:
            mels_val[lang].append((name, mel))
        else:
            mels_train[lang].append((name, mel))

    write_hdf5(GMU_PROC_OUT_FILE_T, mels_train)
    write_hdf5(GMU_PROC_OUT_FILE_VAL, mels_val)


if __name__ == "__main__":
    preprocess_GMU()
