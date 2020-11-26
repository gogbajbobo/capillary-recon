# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt

# %%
group_name = 'Reconstruction'
capillary_recon_path = '/Users/grimax/Desktop/tmp/capillary/tomo_rec.glass_capillary(Mo_mono_40-40).h5'

with h5py.File(capillary_recon_path) as capillary_recon:
    capillary_recon_image = np.array(capillary_recon[group_name])
    print(capillary_recon_image.shape)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(capillary_recon_image[0, :, :])
axes[1].imshow(capillary_recon_image[1030, :, :])
axes[2].imshow(capillary_recon_image[-1, :, :])

# %%
capillary_recon_image_cut = capillary_recon_image[80:, 280:450, 320:530]
print(capillary_recon_image_cut.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(capillary_recon_image_cut[0, :, :])
axes[1].imshow(capillary_recon_image_cut[1000, :, :])
axes[2].imshow(capillary_recon_image_cut[-1, :, :])

# %%
capillary_recon_cut_path = '/Users/grimax/Desktop/tmp/capillary/tomo_rec.glass_capillary(Mo_mono_40-40)_cut.h5'

with h5py.File(capillary_recon_cut_path, mode='w') as file:
    file.create_dataset(group_name, data=capillary_recon_image_cut, compression='lzf')


# %%
