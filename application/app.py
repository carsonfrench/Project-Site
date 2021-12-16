#!/usr/bin/env python
# coding: utf-8
# %%
from io import BytesIO
import json
import requests

import streamlit as st

import torch
from fastai.vision.all import PILImage
from torchvision.models import alexnet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from models_arch import Generator
import matplotlib.pyplot as plt
import os
from PIL import Image



# %%
st.title('Poster Generator')

# %%
nz = 100
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# %%
netG = Generator(ngpu).to(device)
netG_params = torch.load("model1.pth")
netG.load_state_dict(netG_params)

    

# %%
if st.button('Click to generate poster'):
    st.write('Generating poster....')
    netG.eval()
    test_in = torch.randn(1, nz, 1, 1, device=device)
    fake_results = netG(test_in).detach().cpu()
    img = ToPILImage()(fake_results[0])

    st.image(img, width=256, caption='generated movie poster')

# %%

# %%
