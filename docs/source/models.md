# Model Weights

There are several checkpoint files in the `model_weights` directory that can be used with HBDesigner or you can supply your own. You can use the `design_model`, `design_model_ckpt`, or `packing_model_ckpt` command line options to specify which checkpoint file you want to use for the various operations HBDesigner performs.

Here we will briefly describe the various checkpoint files and their uses. 

- `design_020.pt`: Default design checkpoint. It is the high-noise option for the two provided design models. It is best for sampling small networks (2-3 residues) and can give greater sample diversity.
- `design_002.pt`: Low-noise model and is best for large (4+ residues) networks and is more precise than `design_020.pt`.
- `pack.pt`: Provided model for packing calculations. 
- `pippack_model_x_ckpt.pt`: These three models were used for benchmarking purposes and are not recommended for general use.

Don't worry about the YAML files!
