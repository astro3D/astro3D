"""
This file contains an example for running the `convert_indices` and
`write_lhalo_binary` functions within`converter.py`.  Refer to the
documentation of those function for full explanation of each variable.

The default parameters are chosen to match the ASTRO3D Genesis trees as
produced by VELOCIraptor + Treefrog.
"""
#!/usr/bin:env python
from __future__ import print_function
from genesis.utils import treefrog_to_lhalo as frog_to_l


if __name__ == '__main__':

    fname_in = "/fred/oz004/jseiler/genesis/treefrog_trees/new_genesis_version_lhalo_indices.hdf5"
    fname_out = "/fred/oz004/jseiler/genesis/treefrog_trees/test"
    haloID_field = "ID"
    forestID_field = "ForestID"
    Nforests = 10
    write_binary_flag = 0

    frog_to_l.treefrog_to_lhalo(fname_in, fname_out,
                                haloID_field, forestID_field, Nforests,
                                write_binary_flag)
