# genesis/utils

This directory contains a number of useful tools and utilities for handling the data from the
ASTRO3D Genesis simulations.  We have provided a number of example scripts to
run these tools in the `run_scripts` directory.

# forest_sorter

This utility takes input HDF5 merger trees that have not been saved in any specific order and sorts
them on a number of fields; usually an at least an ID field and a mass field.  The user can specify 
on which fields  the sorting should occur. In the default case, the trees are sorted first on the 
`ForestID`, then the `hostHaloID` and finally in descending order of 200mean mass of the halo.  
This results in halos that are sorted in ascending order according to their ForestID, then within 
each Forest, all halos within a single FoF group are grouped and finally within each
FoF-group, the most massive halo is sorted first.

## Tests

First please run the basic tests on the default test data provided by invoking `pytest`.  If this
default test does not pass, please email jseiler@swin.edu.au 

Included in the main function of `tests/forest_sorter_test.py` is an example of
running tests with customized settings.  Of particular note is the `gen_data`
variable.  If this is set to `1`, a small set of sorted trees will be generated
from the specified unsorted HDF5 trees. The number of halos tested on is
handled by the `NHalos_test` variable. 

If you wish to test your fully sorted trees after running `forest_sorter()`, 
set `gen_data=0`, `fname_in` to the path of the **original unsorted** trees and 
`fname_out` to the path of the **sorted** trees. 

If the default test passes but your specific test fails please ensure that your data file is not
corrupt.  Importantly, check that the snapshot keys are named appropriately.  We require the
snapshot fields to include the word **snap** (case insensitive) and assume that the snapshot number
corresponding to the snapshot key is included as a single cluster towards the end of the key;
**snap53_04** should correspond to snapshot number 04 for example. 

**If the snapshot fields are named correctly and your data can be otherwise read in via `h5py`, please
email jseiler@swin.edu.au**

# convert_indices 

This utility takes the sorted HDF5 trees from `forest_sorter` and adjusts the
halo IDs to match the requirements of LHalo trees.  LHalo trees requires that
these IDs are tree local and are the indices of the halos (rather than unique
temporal IDs).

The resulting LHalo compatible trees are saved as a HDF5 file with all other
fields identical to the input Treefrog trees. 

# treefrog_to_lhalo

This function takes the Treefrog trees with the LHalo corrected indices (from
`convert_indices()`) and writes LHalo Tree binary/s. These binary files are of
the format:

* 32-bit integer: `NTrees`, describing the number of trees in the file,
* 32-bit integer: `TotNHalos`, describing the total number of halos within the
file,
* `NTrees` 32-bit integers: `TreeNHalos`, describing the number of halos within each
tree.

Following this header is `TotNHalos` halo entries with data format:

* `Descendant`,          32-bit integer,
* `FirstProgenitor`,     32-bit integer,
* `NextProgenitor`,      32-bit integer,
* `FirstHaloInFOFgroup`, 32-bit integer, 
* `NextHaloInFOFgroup`,  32-bit integer, 
* `Len`,                 32-bit integer,
* `M_Mean200`,           32-bit float,
* `Mvir`,                32-bit float,
* `M_TopHat`,            32-bit float, 
* `Posx`,                32-bit float,
* `Posy`,                32-bit float,
* `Posz`,                32-bit float,
* `Velx`,                32-bit float, 
* `Vely`,                32-bit float, 
* `Velz`,                32-bit float, 
* `VelDisp`,             32-bit float, 
* `Vmax`,                32-bit float, 
* `Spinx`,               32-bit float, 
* `Spiny`,               32-bit float, 
* `Spinz`,               32-bit float, 
* `MostBoundID`,         64-bit integer, 
* `SnapNum`,             32-bit integer, 
* `Filenr`,              32-bit integer,
* `SubHaloIndex`,        32-bit integer, 
* `SubHalfMass`,         32-bit integer.


See [LHaloTreeReader](https://github.com/manodeep/LHaloTreeReader) for an
overview of the LHalo Tree merger pointers.

The function is MPI compatible and the number of final number of files written
is equivalent to the number of processors used to call the function.  These
files are load balanced such that each one will have a similar number of halos
(but not necessarily number of trees).  For example,

```
$ mpirun -np 4 python run_scripts/run_treefrog_to_lhalo.py
```

Would generate 4 LHalo Tree binary files. 
