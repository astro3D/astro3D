"""
This file contains an example for running the `forest_sorter` function within
`forest_sorter.py`.  Refer to the documentation of that function for full
explanation of each variable. 

The default parameters are chosen to match the ASTRO3D Genesis trees as 
produced by VELOCIraptor + Treefrog.
"""

#!/usr/bin:env python
from __future__ import print_function
import numpy as np

from genesis.utils import forest_sorter as fs

if __name__ == '__main__':

    fname_in="/fred/oz004/jseiler/genesis/treefrog_trees/new_genesis_version.hdf5"
    fname_out="/fred/oz004/jseiler/genesis/treefrog_trees/new_genesis_version_sorted.hdf5" 
    haloID_field="ID"
    sort_fields=["ForestID", "hostHaloID", "Mass_200mean"]
    sort_direction=np.array([1,1,-1])
    ID_fields=["Head", "Tail", "RootHead", "RootTail", "ID", "hostHaloID"]
    index_mult_factor=1e12

    forest_sorter(fname_in, fname_out, haloID_field,
                  sort_fields, ID_fields,
                  index_mult_factor, sort_direction) 
