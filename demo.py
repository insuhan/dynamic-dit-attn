import os
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
prompt = "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

# 1. PIL -> images
frames_dir = "./frames"
img1 = video[0]
img1.save(os.path.join(frames_dir, "frame_000.png"))
for fid in range(1, len(video)):
    img2 = video[fid]
    img2.save(os.path.join(frames_dir, "frame_%03d.png" % fid))

# 2. images -> video
save_fps = 24
ofn = "output.mp4"
os.system(f"ffmpeg  -y -loglevel error -f image2 -r {save_fps} -i {frames_dir}/frame_%03d.png -qmin 1 -q:v 1 -pix_fmt yuv420p {ofn}")
import pdb; pdb.set_trace();