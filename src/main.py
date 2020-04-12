from utils import skip_run
from data.data_utils import *
from data_GMU import *

with skip_run('run', 'Get GMU acc data') as check, check():
    extract_GMU()

with skip_run('run', 'Create GMU acc data mels, save as hdf') as check, check():
    preprocess_GMU()

