## Explanations
- `rs`: ratio of reference patches to sample at each step.
- `T`: number of (linear) discretization steps between 0 and 1 to solve the flow ODE.
- `k`: number of top closest patch used to approximate the velocity field.
- `octaves`: number of dyadic scales used for the synthesis.
- `renoise`: "time" $t_r$ used renoise the smooth upsampled image at each resolution: $x_{start}=t_r*x_{upsampled}+(1-t_r)z$. In particular, $t_r=1$ means  additional noise.

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

> If changing the parameters causes a computation error, and you cannot fix the issue, load an example (this resets the parameters to their default values in the case of example 1).