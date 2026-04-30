# Model Weights

There are several checkpoint files in the `model_weights` directory to choose from and you can supply your own. You can use the `design_model`, `design_model_ckpt`, or `packing_model_ckpt` command line options to specify which checkpoint file you want to use.

Here we will briefly describe the various checkpoint files and their uses. 

- `design_020.pt`: Default design checkpoint. It is the high-noise option for the two provided design models. It is best for sampling small networks (2-3) residues and can give greater sample diversity.
- `design_002.pt`: Low-noise model and is best for large (4+ residue) networks and is more precise than `design_020.pt`.
- `pack.pt`: Provided model for packing calculations. 
- `pippack_model_x_ckpt.pt`: These three models were used for benchmarking purposes and are not recommended for general use.