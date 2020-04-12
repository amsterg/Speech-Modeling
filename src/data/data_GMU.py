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
from data_utils import preprocess, mel_spectogram, structure
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
GMU_PROC_OUT_FILE = os.path.join(PROC_DATA_DIR,
                                 'gmu_{}.hdf5'.format(GMU_ACCENT_COUNT))

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
    count = 0
    fnames = speakers_info['filename'].tolist()[count:]
    langs = speakers_info['native_language'].tolist()[count:]
    names = speakers_info['name'].tolist()[count:]

    mel_spects_ = {lang: [] for lang in langs}
    for name, fname, lang in tqdm(zip(names, fnames, langs),
                                  total=len(langs),
                                  bar_format="{l_bar}%s{bar}%s{r_bar}" %
                                  (Fore.GREEN, Fore.RESET)):
        try:
            aud = preprocess(fname)
        except AssertionError as e:
            print("Coudn't process ", len(aud), fname)

        # file_out_ = fname.split('.')[0].replace(
        #     RAW_DATA_DIR, INTERIM_DATA_DIR) + '_' + AUDIO_WRITE_FORMAT

        # soundfile.write(file_out_, aud, SAMPLING_RATE)
        
        mel = mel_spectogram(aud)

        mel_spects_[lang].append((name, mel))


    gmu_proc_file = h5py.File(GMU_PROC_OUT_FILE, 'w')
    for lang in mel_spects_:
        lang_group = gmu_proc_file.create_group(lang)
        for mel in mel_spects_[lang]:
            lang_group.create_dataset("mel_spects_{}".format(mel[0]),
                                      data=mel[1])

    gmu_proc_file.close()


if __name__ == "__main__":
    preprocess_GMU()
