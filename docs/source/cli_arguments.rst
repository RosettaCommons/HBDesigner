Command Line Options
====================

Usage Advice
------------

At larger ``--n_res``, packing is harder, so you will get fewer good designs per ``--n_samples``. This means you might want to increase ``--n_samples`` when increasing ``--n_res``. Here is a good place to start:

.. code-block:: bash

    --n_res=2, --n_samples=100
    --n_res=3, --n_samples=200
    --n_res=4, --n_samples=500
    --n_res=5, --n_samples=500
    --n_res=6, --n_samples=1000

Smaller amino acids, especially SER and THR, have notably higher success rates. This means that, if you don't care what amino acids are in your network, you can get higher success rates using ``--guide_seq SXX``, ``--guide_seq TXX``, etc.

.. argparse::
    :module: hbdesigner.inference.parsers
    :func: get_hbdes_parser