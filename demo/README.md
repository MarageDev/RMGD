---
title: Nifty
emoji: 🌘
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
python_version: 3.13.7
app_file: app.py
pinned: false
license: mit
short_description: Non-parametric flow-matching model w/ non-local patch match
---

<h1 style="text-align:center;">Official pytorch implementation of the paper: "NIFTY: A non-local image flow matching for texture synthesis"</h1>
<div style="text-align:center;">
  
[Pierrick Chatillon](https://scholar.google.com/citations?user=8MgK55oAAAAJ&hl=en) | [Julien Rabin](https://sites.google.com/site/rabinjulien/) | [David Tschumperlé](https://tschumperle.users.greyc.fr/) | [Mahé Duval](https://github.com/MarageDev)

</div>
<div style="text-align:center;">
  
[Arxiv](http://arxiv.org/abs/2509.22318) [HAL](https://hal.science/hal-05287967)

</div>

## Overview

NIFTY is a non-parametric flow-matching model built on non-local patch matching, which avoids the need for neural network training while alleviating common shortcomings of patch-based methods, such as poor initialization or visual artifacts. 
<div style="display:flex; justify-content:center;"> 
  
  ![Alt text](results/layer1.jpg)
  ![Alt text](results/layer2.jpg)
  ![Alt text](results/layer3.jpg)
</div>



## Info

This is a Hugging Face hosted demo of the one from [Github](https://github.com/PierrickCh/Nifty), the CPU only version. The repository contains another demo to compare Nifty to trained (and trainable) UNets, but it requires a GPU to have a reasonnable computation time for the training.

More information regarding this algorithm can be found here : 
- [Arxiv](http://arxiv.org/abs/2509.22318) 
- [HAL](https://hal.science/hal-05287967)
- [Github](https://github.com/PierrickCh/Nifty)

## Demo
The `Debug Mode` lets you visualize the copied regions, highlighted in gray, as well as the newly generated ones.

The input `Height` and `Width` parameters (on the left) let you resize the image if needed, in order to reduce computation time.

The output `Height` and `Width` parameters (on the right) let you define the size of the texture synthesis to be generated.

Clicking `Generate` starts the texture synthesis process with the specified parameters.

Clicking `Clear CUDA Cache` clears the GPU cache used by the program; use this when an `Out Of Memory` error occurs.

If changing the parameters causes a computation error, and you cannot fix the issue, load an example (this resets the parameters to their default values in the case of example 1).

Parameters:
- `rs`: ratio of reference patches to sample at each step.
- `T`: number of discretization steps used to solve the flow matching ODE.
- `k`: number of nearest patches used to approximate the field velocity (flow matching method).
- `octaves`: number of dyadic scales used for the synthesis.
- `renoise`: factor used to adjust the intensity of the noise added at each step when the resolution increases.

- `Blend`: mixes the synthesized image with the input image, which can help preserve part of the input image structure.
- `Blend Alpha`: weighting factor for the mix between the synthesized texture and the input image.
- `Blend Map`: if checked, textures will be blended linearly (from right to left, with a mix of both in the middle).

- `Patch Size`: size of the patches used by the algorithm (the larger the patches, the larger the copied areas).
- `Stride`: number of jumps used to compute flow matching (increasing the stride reduces computation time).
- `Warmup` (if `Memory` is enabled): number of initial steps during which the flow is not applied, which can help stabilize the synthesis at the beginning.
- `Memory`: use the memory-efficient version of Nifty, which does not store all intermediate synthesized images during flow integration, but only the current image.
- `Seed`: random seed (the same seed for a given random process returns the same value; thus, for texture synthesis, the same seed gives the same result if the parameters are identical).
- `Noise`: adds noise during synthesis, which can help escape local minima and produce more diverse results.
- `Spot Size`: size of the spots used for synthesis, relative to the patch size.

A list of examples from the paper is available at the bottom of the demo.

## Acknowledgments
This  work  was  partly  funded  by  the  Normandy  Region  through  the IArtist excellence label project.

## Citation
If you use this code for your research, please cite our paper, ICASSP 2026 citation will be updated after publication:

```
@inproceedings{NIFTY,
  TITLE = {{NIFTY: a Non-Local Image Flow Matching for Texture Synthesis}},
  AUTHOR = {Chatillon, Pierrick and Rabin, Julien and Tschumperl{\'e}, David},
  URL = {https://hal.science/hal-05287967},
  BOOKTITLE = {{ICASSP}},
  ADDRESS = {Barcelona, Spain},
  YEAR = {2026},
  MONTH = May,
  DOI = {10.48550/arXiv.2509.22318},
  KEYWORDS = {Machine Learning (cs.LG) ; Computer Vision and Pattern Recognition (cs.CV) ; Flow Matching ; Texture synthesis ; Image synthesis ; Generative model},
  PDF = {https://hal.science/hal-05287967v1/file/2509.22318v1.pdf},
  HAL_ID = {hal-05287967},
  HAL_VERSION = {v1},
}

```

## License
This work is under the MIT license.

## Disclaimer
The code is provided "as is" with ABSOLUTELY NO WARRANTY expressed or implied.