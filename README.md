# RMGD
Réinterprétation des modèles génératifs de diffusion
## Content
The algorithms under `./algorithms` are python scripts of algorithms described in [1] and flow matching variants with other features. The algorithm were developped in a linear way, so each version number is almost always equal to its predecessor with some improvements or new features.
Here's a list of the algorithm and a short description : 

<table>
  <thead>
    <tr>
      <th width="200px">File</th>
      <th width="1500px">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th><code>algo1.py</code></th>
      <td>Implementation of the algorithm 1 from [1] (see the paper for in depth explanation)</td>
    </tr>
    <tr>
      <th><code>algo2.py</code></th>
      <td>Implementation of the algorithm 2 from [1] (see the paper for in depth explanation)</td>
    </tr>
    <tr>
      <th><code>algo3.py</code></th>
      <td>Implementation of the algorithm 3 from [1] (see the paper for in depth explanation)</td>
    </tr>
    <tr>
      <th><code>algo4.py</code></th>
      <td>Algorithm 3 similar to NIFTY with a random gaussian noise initialisation</td>
    </tr>
    <tr>
      <th><code>algo5.py</code></th>
      <td>Patch based flow matching on a multiple image dataset (BxCxHxW) in the RGB space</td>
    </tr>
    <tr>
      <th><code>algo6.py</code></th>
      <td>Algorithm 5 with multi-scale (initial image dataset tensor with bicubic scale interpolation)</td>
    </tr>
    <tr>
      <th><code>algo7.py</code></th>
      <td>Same as algorithm 6 but this time with pre-loaded batch tensors for the different scales (no interpolation, retains details)</td>
    </tr>
    <tr>
      <th><code>algo8.py</code></th>
      <td>Algorithm 7 with some optimization and stride</td>
    </tr>
    <tr>
      <th><code>algo9_cov.py</code></th>
      <td>Algorithm 8 with covariance matrix to retain details and general shapes + gaussian noise initialisation with the calculated covariance (with Julien Rabin and Pierrick Chatillon)</td>
    </tr>
    <tr>
      <th><code>algo10.py</code></th>
      <td>Algorithm 9 but with some fixes</td>
    </tr>
  </tbody>
</table>

Here's a list of the other algorithms from the latent space "timeline" : 

<table>
  <thead>
    <tr>
      <th width="200px">File</th>
      <th width="1500px">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th><code>algo5_latent.py</code></th>
      <td>Algorithm 5 but in the latent space of TAESD</td>
    </tr>
    <tr>
      <th><code>algorithm_PCAGMM_latent.py</code></th>
      <td>Main final algorithm based on algorithm 5, in latent space, to initialise in combination with other scripts (see below)</td>
    </tr>
  </tbody>
</table>

## Demo
The repository contains a general demo file using the open-source Python package Gradio to render the user interface. The main demo file is located under : `demo/app.py`.
You can also find a demo hosted on [Hugging Face](https://huggingface.co/spaces/Marage/Nifty) which contains a demo an algorithm in this repository as well as one of the [NIFTY algorithm](https://github.com/PierrickCh/Nifty).

### How to run
To launch the demo, start the python script in the virtual environment :  
```shell
python ./demo/app.py
```
or use gradio hot reload mode (if you plan to edit the code) with 
```shell
gradio ./Demos/app.py
```
### How to load custom dataset
The main loading and encoding file for the datasets is `./dataset_chunk_loader_encoder.py`.
> Note that for now, and ease of use, it's only handling `.jpg` and `.jpeg` files, but modifying to other formats should be straightforward. If the dataset you're trying to load and encode is using another format (like `.parquet` or MNIST like format, `./others/dataset_loader.py` contains other loading functions

The main parameters are exposed as global variables, they are the following :
- `root` : path under which the encoded dataset will be saved
- `image_source` : path of the directory where the images are stored in
- `image_size` : defines the output size of the images (squared shape)
- `batch_size` : number of batches to make when processing the images, keep at 1 if the model used is really fast to encode (otherwise the CPU will get flooded and it will be slower)
- `num_workers` : how many subprocesses to use for data loading

> In this case, it's using [TAESD](https://github.com/madebyollin/taesd) to encode the images

# Références
1 
```
Duval, Denis, and Agnès Desolneux.
"Réinterprétation des modèles génératifs de diffusion."
30e Colloque sur le traitement du signal et des images.
2025.
```

2
```
Chatillon, Pierrick, Julien Rabin, and David Tschumperlé.
"NIFTY: a Non-Local Image Flow Matching for Texture Synthesis."
ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
IEEE, 2026.
```
