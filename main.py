# -*- coding: utf-8 -*-
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
from mpl_toolkits import mplot3d

from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from skimage import morphology
from porespy.filters import find_disconnected_voxels

from collections import namedtuple

from stl import mesh as stl_mesh

# %%
capillary_recon_path = '/Users/grimax/Desktop/tmp/capillary/56adacb0-0319-464f-a455-7273fc1fb755.h5'
capillary_recon = h5py.File(capillary_recon_path)
capillary_recon_image = np.array(capillary_recon['Reconstruction'])
print(capillary_recon_image.shape)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(capillary_recon_image[10, :, :])
axes[1].imshow(capillary_recon_image[1000, :, :])
axes[2].imshow(capillary_recon_image[-1, :, :])

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(capillary_recon_image[:, 105, :])
axes[1].imshow(capillary_recon_image[:, :, 85])

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(capillary_recon_image[:50, 105, :])
axes[1].imshow(capillary_recon_image[:50, :, 85])
axes[2].imshow(capillary_recon_image[-50:, 105, :])
axes[3].imshow(capillary_recon_image[-50:, :, 85])

# %%
capillary_recon_image = capillary_recon_image[20:, :, :]
print(capillary_recon_image.shape)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(capillary_recon_image[0, :, :])
axes[1].imshow(capillary_recon_image[1000, :, :])
axes[2].imshow(capillary_recon_image[-1, :, :])

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(capillary_recon_image[:50, 105, :])
axes[1].imshow(capillary_recon_image[:50, :, 85])
axes[2].imshow(capillary_recon_image[-50:, 105, :])
axes[3].imshow(capillary_recon_image[-50:, :, 85])

# %%
# grayscale image binarization
threshold = threshold_otsu(capillary_recon_image)
capillary_binary_image = capillary_recon_image >= threshold

plt.figure(figsize=(15, 5))
plt.hist(np.ravel(capillary_recon_image), bins=256, color='lightgray')
plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)

plt.figure(figsize=(15, 5))
plt.hist(np.ravel(capillary_recon_image), bins=256, color='lightgray', log=True)
plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)

print(threshold)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(capillary_binary_image[0, :, :])
axes[1].imshow(capillary_binary_image[1000, :, :])
axes[2].imshow(capillary_binary_image[-1, :, :])

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(capillary_binary_image[:, 105, :])
axes[1].imshow(capillary_binary_image[:, :, 85])

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(capillary_binary_image[:50, 105, :])
axes[1].imshow(capillary_binary_image[:50, :, 85])
axes[2].imshow(capillary_binary_image[-50:, 105, :])
axes[3].imshow(capillary_binary_image[-50:, :, 85])

# %%
levitating_stones = find_disconnected_voxels(capillary_binary_image)
print(np.sum(levitating_stones))

# %%
# remove levitating stones
capillary_binary_image[levitating_stones] = False


# %%
def show_3d_image(image):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.voxels(image)


def mesh_calculation(image, pad_width=0, constant_values=True, level=None):

    if pad_width:
        image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=constant_values)

    verts, faces, norm, val = marching_cubes(image, level=level)
    result = namedtuple('mesh', ('verts', 'faces', 'norm', 'val'))
    result.verts = verts - pad_width/2
    result.faces = faces
    result.norm = norm
    result.val = val

    return result


def show_mesh(mesh):
    fig = plt.figure()
    axes = mplot3d.Axes3D(fig)
    mesh_fig = mplot3d.art3d.Poly3DCollection(mesh.verts[mesh.faces])
    axes.add_collection3d(mesh_fig)

    scale = mesh.verts.flatten('F')
    axes.auto_scale_xyz(scale, scale, scale)

    
def save_mesh_to_stl(mesh, filename):
    mesh_reg = list(map(lambda x: (0, x.tolist(), 0), mesh.verts[mesh.faces]))
    mesh_reg = np.array(mesh_reg, dtype=stl_mesh.Mesh.dtype)
    mesh_reg = stl_mesh.Mesh(mesh_reg, remove_empty_areas=True)
    mesh_reg.save(filename)



# %%
# test mesh calculation on small fragment of binarized image
im = capillary_binary_image[990:1010, :, :]

# %%
# use only 1 row of voxels to increase rendering speed
show_3d_image(im[0:1, :, :])

# %%
mesh = mesh_calculation(im)
show_mesh(mesh)
save_mesh_to_stl(mesh, 'mesh_bin_0.stl')

# %%
mesh = mesh_calculation(im, pad_width=1)
show_mesh(mesh)
save_mesh_to_stl(mesh, 'mesh_bin_1.stl')

# %%
mesh = mesh_calculation(~im, pad_width=1)
show_mesh(mesh)
save_mesh_to_stl(mesh, 'mesh_bin_2.stl')

# %%
# test mesh calculation on small fragment of grayscale image
im = capillary_recon_image[990:1010, :, :]

# %%
mesh = mesh_calculation(im, level=threshold)
show_mesh(mesh)
save_mesh_to_stl(mesh, 'mesh_recon_0.stl')

# %%
mesh = mesh_calculation(im, pad_width=1, level=threshold)
show_mesh(mesh)
save_mesh_to_stl(mesh, 'mesh_recon_1.stl')

# %%
mesh = mesh_calculation(-im, pad_width=1, level=-threshold)
show_mesh(mesh)
save_mesh_to_stl(mesh, 'mesh_recon_2.stl')

# %%
# calculate and save meshes for whole object
mesh = mesh_calculation(capillary_binary_image)
save_mesh_to_stl(mesh, 'mesh_whole_bin_0.stl')

mesh = mesh_calculation(capillary_binary_image, pad_width=1)
save_mesh_to_stl(mesh, 'mesh_whole_bin_1.stl')

mesh = mesh_calculation(~capillary_binary_image, pad_width=1)
save_mesh_to_stl(mesh, 'mesh_whole_bin_2.stl')

mesh = mesh_calculation(capillary_recon_image, level=threshold)
save_mesh_to_stl(mesh, 'mesh_whole_recon_0.stl')

mesh = mesh_calculation(capillary_recon_image, pad_width=1, level=threshold)
save_mesh_to_stl(mesh, 'mesh_whole_recon_1.stl')

mesh = mesh_calculation(-capillary_recon_image, pad_width=1, level=-threshold)
save_mesh_to_stl(mesh, 'mesh_whole_recon_2.stl')

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(capillary_binary_image[0, :, :])
axes[1].imshow(capillary_binary_image[1000, :, :])
axes[2].imshow(capillary_binary_image[-1, :, :])


# %%
def closing_3d(image, element):
    result = np.zeros(image.shape, dtype=np.bool)
    for i in range(image.shape[0]):
        result[i, :, :] = morphology.binary_closing(image[i, :, :], element)
    return result


def opening_3d(image, element):
    result = np.zeros(image.shape, dtype=np.bool)
    for i in range(image.shape[0]):
        result[i, :, :] = morphology.binary_opening(image[i, :, :], element)
    return result



# %%
# fill the capillary central hole 
closing_element = morphology.disk(10)
closing_image = closing_3d(capillary_binary_image, closing_element)
plt.imshow(closing_image[0, :, :])

# %%
# get the capillary central hole
diff_image = closing_image != capillary_binary_image
plt.imshow(diff_image[0, :, :])

# %%
# find and erase the remaining garbage
opening_element = morphology.disk(1)
opening_image = opening_3d(diff_image, opening_element)
plt.imshow(opening_image[0, :, :])

# %%
# check levitating stones and closed pores
# if we have not null number of stones/pores — we should make a correction in closing/opening elements size
levitating_stones = find_disconnected_voxels(opening_image)
print(f'levitating_stones: {np.sum(levitating_stones)}')
closed_pores = find_disconnected_voxels(~opening_image)
print(f'closed_pores: {np.sum(closed_pores)}')

# %%
