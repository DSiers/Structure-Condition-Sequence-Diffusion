# Structure-Condition-Sequence-Diffusion

Compressed version of my Master Thesis Codebase

## Project Structure

```sh
├───ControlVideo                                            # ControlVideo - Adjusted for more recent diffusers version
│   └───checkpoints                                         # https://github.com/YBYBZhang/ControlVideo
│       ├───sd-controlnet-depth
│       ├───sd-controlnet-depth_v1.1
│       ├───sd-controlnet-hed
│       ├───sd-controlnet-hed_v1.1
│       └───stable-diffusion-v1-5_dreambooth_taichi_hd
├───src                                                     # Base Implementation: https://github.com/CompVis/stable-diffusion
│   hed_utils.py                                            # HED utils for extraction/preprocessing
│   midas_utils.py                                          # MiDaS utils for depth estimation and preprocessing
│   util.py                                                 # general utils
│   __init__.py
│
├───configs
├───data                                                    # Datasets
│   │   cityscape.py
│   │   kinetics.py
│   │   kinetics_stream.py
│   │   taichi.py
│   │   ucf.py
│   │
│   └───prompts                                             # Generated Prompts used for evaluation
│           prompts_cityscapes.json
│           prompts_kinetics.json
│           prompts_taichi.json
│
├───inference                                               # Inference Class -- new
│       inference_script.py                                 # Only supports VAE sampling yet
│       inference_utils.py
│       __init__.py
│
├───models
│    │   autoencoders.py                                    # SD Autoencoder Base
│    │   autoencoders_cond.py                               # Autoencoder for Depth and HED maps
│    │ 
│    │
│    └───modules
│        │   attention.py                                   # Base Attention + Various temporal extensions
│        │   distributions.py                               # Adjusted VAE sampling for sequences
│        │   ema.py                                         
│        │   __init__.py
│        │
│        ├───diffusion
│        │   │   ddim.py                                    
│        │   │   ddpm.py                                    # DDPM base from SD + 2D and 3D extensions for Structure Condition generation
│        │   │   sampling_util.py
│        │   │   __init__.py
│        │   │
│        │   └───dpm_solver                                 
│        │           dpm_solver.py
│        │           sampler.py
│        │           __init__.py
│        │
│        └───diffusionmodules                               
│                model.py
│                openaimodel_new.py                         # Implementation for the temporal UNet with context masking
│                util.py                                    # Ported Various TC metric Implementations to TorchMetrics + TC_Loss + some helper functions
│                __init__.py
├───taichi_model                                            # Best performing TaiChi model checkpoint
│   ├───checkpoints
│   ├───configs
│   └───stable-diffusion-v1-5_dreambooth_taichi_hd
│       └────unet
└───TC_env                                                  # Old repo to evaluate temporal consistency
```

## Autoencoder for Structure Conditions

We provide a checkpoint for a VAE that compresses 512x512 depth images into 32x32 latents.

![](https://github.com/DanielSiersleben/Structure-Condition-Sequence-Diffusion/blob/main/viz/VAE_recon.png)

## Structure Condition Sequence Diffusion

| Generated Depth Sequence  | Generated Video with ControlVideo |
| ------------- | ------------- |
| ![](https://github.com/DanielSiersleben/Structure-Condition-Sequence-Diffusion/blob/main/viz/gifs/cond_0.gif)  | ![](https://github.com/DanielSiersleben/Structure-Condition-Sequence-Diffusion/blob/main/viz/gifs/video_0.gif)  |
