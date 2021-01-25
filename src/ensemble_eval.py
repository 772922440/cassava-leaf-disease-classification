from os.path import join
import os
from pathlib import Path
from contextlib import contextmanager
from scipy.special import softmax
import pandas as pd
from tqdm.auto import tqdm
import torch
import warnings 
warnings.filterwarnings('ignore')

# our codes
from utils import utils, torch_utils
from dataset import leafdisease as ld
from model import get_backbone

# read config
config = utils.read_all_config()
utils.mkdir(config.output_dir)
torch_utils.seed_torch(seed=config.seed)

# test data
test = pd.read_csv(join(config.data_base_path, config.test_csv))
test['filepath'] = test.image_id.apply(lambda x: join(config.data_base_path, config.test_images, f'{x}'))

# transform
transforms = ld.get_transform(image_size=config.image_size)[1]
test_dataset = ld.CLDDataset(test, 'test', transform=transforms)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)


def inference(model_list, test_loader, device):
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []

        # load models
        for m in range(config.model_list):
            backbone = m['backbone']
            model = get_backbone(backbone, config).to(device=config.device)
            model.eval()

            for filename in m['filename']:
                model_path = os.path.join(config.model_base_path, backbone, filename)
                model.load_state_dict(torch.load(model_path))

                with torch.no_grad():
                    y_preds = model(images)
                avg_preds.append(y_preds.softmax(dim=-1).to('cpu'))

        # simple mean weights
        avg_preds = torch.mean(avg_preds, dim=0)
        probs.append(avg_preds)

    probs = torch.cat(probs, dim=0)
    return probs

# predict
probs = inference(config.model_list, test_loader, config.device)
test['label'] = torch.argmax(probs, dim=-1).numpy()
test[['image_id', 'label']].to_csv(os.path.join(config.output_dir, 'submission.csv'), index=False)
test.head()