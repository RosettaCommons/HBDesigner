HBDesigner Examples
===================

Several examples of how to use HBDesigner have been included in this repository, you can find them in the `examples` folder. Descriptions of these examples can be found below. 

Running the Examples
--------------------

.. important:: 
    It is assumed that you will be running these examples in their respective subdirectories in the `examples` folder. The examples use relative paths to access the necessary files, so if you run them from a different location you will need to adjust the paths accordingly.

    These examples also assume that you have access to 8 CPU nodes to use as 'workers' for HBDesigner. If you have access to fewer, then you will need to adjust the values of `n_workers` in the shell scripts before running. 

If you have installed HBDesigner using uv, conda, mamba, or pip you can run these examples via 

.. code-block:: bash

    ./<example_script>.sh

(Though you will need to make sure you have the appropriate environment activated if you installed via conda/mamba/uv).

However, if you installed HBDesigner via pixi, you will need to run the examples via

.. code-block:: bash

    pixi run --manifest-path=<relative path to>/pyproject.toml <example_script>.sh

.. toctree::
   :maxdepth: 1
   :caption: Explanation of the Examples:

   unconditional_monomer_design.md
   interface_design.md
   symmetric_design.md
   sequence_conditioning.md
   virtual_guide_atom_conditioning.md

For more information about the options used in these examples, see :doc:`../cli_arguments`.