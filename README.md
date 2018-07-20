# genesis/utils

This directory contains a number of useful tools and utilities for handling the data from the
ASTRO3D Genesis simulations.  We have provided a number of example scripts to
run these tools in the `run_scripts` directory.

# Forest Sorter

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
set `gen_data=0`, `fname_in` to the path of the *original unsorted* trees and 
`fname_out` to the path of the *sorted* trees. 

If the default test passes but your specific test fails please ensure that your data file is not
corrupt.  Importantly, check that the snapshot keys are named appropriately.  We require the
snapshot fields to include the word **snap** (case insensitive) and assume that the snapshot number
corresponding to the snapshot key is included as a single cluster towards the end of the key;
**snap53_04** should correspond to snapshot number 04 for example. 

**If the snapshot fields are named correctly and your data can be otherwise read in via h5py, please
email jseiler@swin.edu.au**

# converter 

This utility takes the sorted HDF5 trees from `forest_sorter` and adjusts the
halo IDs to match the requirements of LHalo trees.  That is, the function
`convert_indices()` converts the IDs to ones that are local within each forest.  
The function `write_LHalo_binary()` then writes these adjusted trees to a
binary file with the LHalo tree structure, designed to be read by Semi-Analyti
Models such as [SAGE](https://github.com/darrencroton/sage). 
