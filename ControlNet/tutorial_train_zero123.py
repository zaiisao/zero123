import sys

sys.path.append("../zero123")

from share import *

import pytorch_lightning as pl
from einops import rearrange
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from ldm.data.simple import ObjaverseData
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = '../ControlNet/models/control_zero123_105000.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
image_transforms_size = 256

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_zero123.yaml').cpu() # JA: model is an object of ControlLDM
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Image transforms
image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_transforms_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
])

# Misc
dataset = ObjaverseData(
    "/home/jaehoon/.objaverse/hf-objaverse-v1/views_whole_sphere",
    total_view=4, validation=False, image_transforms=image_transforms)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
