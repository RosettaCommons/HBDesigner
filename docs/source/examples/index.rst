HBDesigner Examples
===================

Several examples of how to use HBDesigner have been included in this repository, you can find them in the ``examples`` folder. Descriptions of these examples can be found below. 

Running the Examples
--------------------

Input structures
^^^^^^^^^^^^^^^^

HBDesigner takes a ``.pdb`` file as its primary input. It can have a sequence and/or sidechains on it, but they will be removed to create a PolyGLY backbone before use.

If you want to design intra-chain or monomer networks, you should include only the chain you wish to design.

If you want to design inter-chain or interface networks, you should include only the chains forming the interface you wish to design.

Output files
^^^^^^^^^^^^
HBDesigner produces two kinds of output:
-  PDB files named "HBDes_rank_N.pdb", where N is the rank calculated by HBDesigner. Lower ranks are "better".
- A CSV file named "HBDes_stats.csv", which includes all of the scores and residue IDs of all designed networks. This includes networks that passed the scoring filters but didn't make the ``--top_k`` cutoff.

HBDesigner will only output networks that pass all of its scoring filters and meet its definition of 'successful'. A 'successful' design meets the following criteria:
- All network residues must be engaged in at least 1 sidechain-sidechain H-bond
- All network residues must form a single contiguous network
- The network passes the minimum thresholds for saturation, BUHs, BUPHs, etc. (see `Scoring parameters`_ below)

After filtering, HBDesigner will rank the remaining networks using various score terms, with the following priority:
1. ``buried_unsat_Hpol`` (buried unsatisfied polar H atoms): the fewer the better.
2. ``saturation``: the fraction of total h-bonding "capacity" that is being used across all network residue sc atoms: the higher the better.
3. HBond Score (``HB_Score_full``): the change in Rosetta energy provided by the designed network, calculated against an identical PolyG backbone: the lower (more negative) the better.

This setup has a few implications:

- It is possible for HBDesigner to return 0 networks, if given a hard enough task and/or few enough tries at it. If this happens, try increasing ``--n_samples``.

- If HBDesigner finds more networks than ``--top_k`` allows, it will only return the ``--top_k`` "best" according to the ranking scheme. If you want more outputs, increase ``--top_k``.

.. important:: 
    It is assumed that you will be running these examples in their respective subdirectories in the ``examples`` folder. The examples use relative paths to access the necessary files, so if you run them from a different location you will need to adjust the paths accordingly.

    These examples also assume that you have access to 8 CPU nodes to use as 'workers' for HBDesigner. If you have access to fewer, then you will need to adjust the values of ``n_workers`` in the shell scripts before running. 

If you have installed HBDesigner using uv, conda, mamba, or pip you can run these examples via 

.. code-block:: bash

    ./<example_script>.sh

(Though you will need to make sure you have the appropriate environment activated if you installed via conda/mamba/uv.)

However, if you installed HBDesigner via pixi, you will need to run the examples via

.. code-block:: bash

    pixi run --manifest-path=<relative path to>/pyproject.toml <example_script>.sh

You can see a full example of this in the ``monomer`` directory.

.. toctree::
   :maxdepth: 1
   :caption: Explanation of the Examples:

   monomer.md
   interface_design.md
   postprocessing.md

For more information about the options used in these examples, see :doc:`../cli_arguments`.