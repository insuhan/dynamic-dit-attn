import os
import time
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import seaborn as sns
from glob import glob
import imageio


def save_figures():

    save_path = "./figures/attn_heatmap/cogvideox5b"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cond_type = "cond"
    size = 128

    tic = time.time()
    for layer_id in range(42):
        attn_scaled = torch.load(f"metadata/layer{layer_id}_attn_{size}_cond.pth", map_location="cpu")
        num_steps = attn_scaled.shape[0]
        for step_id in range(num_steps):
            fname = os.path.join(save_path, f"layer{layer_id}_step{step_id}_cond.png")
            fig, ax = plt.subplots(6, 8, figsize=(18, 12))
            fig.suptitle(f"sora_prompt1, layer: {layer_id}, {cond_type}, step: {step_id}/{50}", fontsize=16)
            for i in range(6):
                for j in range(8):
                    head_id = i*8 + j
                    ax[i][j].set_title(f"head{head_id}")
                    sns.heatmap(
                        attn_scaled[step_id, head_id],
                        annot=False, square=True, cmap="crest", cbar=True,
                        ax=ax[i][j], cbar_kws={"shrink": 0.75},
                        xticklabels=False, yticklabels=False) # vmin=0.0, vmax=1.0,
            fig.tight_layout()
            plt.savefig(fname)
            print(f"layer_id: {layer_id}/42, step: {step_id}, time: {time.time() - tic:.4f} sec")


def png_to_gif():
    layer_id = 0
    cond = 'cond'
    images = []

    output_path = f"./figures/layer{layer_id}_cond.gif"
    input_path = "./figures/attn_heatmap/cogvideox5b"
    paths = sorted(glob(input_path + f"/layer{layer_id}_*"))
    if not len(paths):
        raise ValueError(
            "No images found in save path, aborting (did you pass save_intermediate=True to the generate"
            " function?)"
        )
    if len(paths) == 1:
        print("Only one image found in save path, (did you pass save_intermediate=True to the generate function?)")
    # frame_duration = total_duration / len(paths)
    # durations = [frame_duration] * len(paths)
    # if extend_frames:
    #     durations[0] = 1.5
    #     durations[-1] = 3
    paths.sort(key= lambda x: int(x.split("step")[-1].split("_")[0]))
    images = []
    for file_name in paths:
        if file_name.endswith(".png"):
            images.append(imageio.imread(file_name))
    imageio.mimsave(output_path, images, format='GIF', duration=100, loop=100)
    print(f"gif saved to {output_path}")
    import pdb; pdb.set_trace();

if __name__ == "__main__":
    save_figures()
    # png_to_gif()