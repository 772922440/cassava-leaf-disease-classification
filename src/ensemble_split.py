from os.path import join
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import warnings 
warnings.filterwarnings('ignore')

# our codes
from utils import utils, torch_utils
from dataset import leafdisease as ld

# read config
config = utils.read_all_config()
logger = utils.get_logger(config)
torch_utils.seed_torch(seed=config.seed)

train = pd.read_csv(join(config.data_base_path, config.train_csv))

folds = train.copy()
Fold = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[config.target_col])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
print(folds.groupby(['fold', config.target_col]).size())

folds[['image_id', 'label', 'fold']] \
    .to_csv(join(config.data_base_path, str(config.k_folds) + 'folds.csv'), index=False)