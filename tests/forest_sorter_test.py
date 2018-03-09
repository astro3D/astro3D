#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from optparse import OptionParser
import sys
import h5py
import os
import pytest

from genesis.utils import forest_sorter as fs

def parse_inputs():
    """

    Parses the command line input arguments.
    
    Parameters
    ----------

    None.   
 
    Returns
    ----------

    opt: optparse.Values.  Required.  
        Values from the OptionParser package.  
        Values are accessed through ``opt.Value`` and cast into a dictionary using ``vars(opt)`` 
    """

    parser = OptionParser()
    test_dir = os.path.dirname(__file__)

    parser.add_option("-f", "--fname_in", dest="fname_in", 
                      help="Path to test HDF5 data. Default: {0}/test_data.hdf5".format(test_dir), 
                      default="{0}/test_data.hdf5".format(test_dir))
    parser.add_option("-o", "--fname_out", dest="fname_out", 
                      help="Path to sorted output HDF5 data file. Default: "
                       "{0}/test_sorted.hdf5".format(test_dir),
                      default="{0}/test_sorted.hdf5".format(test_dir))
    parser.add_option("-n", "--NHalos_test", dest="NHalos_test", 
                      help="Minimum number of halos to test using. Default: 10,000", 
                      default = 10000, type = int) 
    parser.add_option("-s", "--sort_id", dest="sort_id", 
                      help="Field name for the key we are sorting on. Default: ForestID.",  
                      default = "ForestID")
    parser.add_option("-m", "--mass_def", dest="sort_mass", 
                      help="Field name for the mass key we are sorting on. Default: Mass_200mean.", 
                      default = "Mass_200mean")
    parser.add_option("-i", "--HaloID", dest="halo_id", 
                      help="Field name for halo ID. Default: ID.", 
                      default = "ID")
    parser.add_option("-p", "--ID_fields", dest="ID_fields", 
                      help="Field names for those that contain non-merger IDs.  Default: ('ID',"
                      "Tail', 'Head').", default = ("ID", "Tail", "Head")) 
    parser.add_option("-x", "--index_mult_factor", dest="index_mult_factor", 
                      help="Conversion factor to go from a unique, per-snapshot halo index to a \
                      temporally unique haloID.  Default: 1e12.", 
                      default = 1e12)

    (opt, args) = parser.parse_args()

    # Print some useful startup info. #
    print("")
    print("Running test functions")
    print("Performing tests on a minimum of {0} halos.".format(opt.NHalos_test))
    print("The HaloID field for each halo is '{0}'.".format(opt.halo_id)) 
    print("Sorting on the '{0}' field.".format(opt.sort_id))
    print("Sub-Sorting on the '{0}' field.".format(opt.sort_mass))
    print("")

    return opt

def my_test_sorted_order(opt):
    """

    Checks the indices of the sorted output file to ensure that the sorting has been performed
    correctly. 

    Parameters
    ----------

    opt: Dictionary.  
        Dictionary containing the option parameters specified at runtime.  
        Used to get file name and sorting fields. 

    Returns
    ----------

    None. ``Pytest.fail()`` is invoked if the test fails. 

    """

    with h5py.File(opt["fname_out"], "r") as f_in:
           
        Snap_Keys, Snap_Nums = fs.get_snapkeys_and_nums(f_in.keys())

        for snap_key in Snap_Keys:
            NHalos = len(f_in[snap_key][opt["halo_id"]])
            if NHalos < 2:  # Skip snapshots that wouldn't be sorted. 
                continue
 
            for idx in range(NHalos - 1):
                halo_id = f_in[snap_key][opt["halo_id"]][idx]
                halo_id_next = f_in[snap_key][opt["halo_id"]][idx + 1]

                outer_sort = f_in[snap_key][opt["sort_id"]][idx]
                outer_sort_next = f_in[snap_key][opt["sort_id"]][idx + 1]

                inner_sort = f_in[snap_key][opt["sort_mass"]][idx]
                inner_sort_next = f_in[snap_key][opt["sort_mass"]][idx + 1]
                
                if (outer_sort > outer_sort_next):
                    print("For Halo ID {0} we had a {1} of {2}.  After sorting via lexsort with "
                          "outer-key {1}, the next Halo has ID {3} and a {1} of {4}".format(halo_id, 
                          opt["sort_id"], outer_sort, halo_id_next, outer_sort_next))
                    print("Since we are sorting using {0} they MUST be in ascending order.".format(opt["sort_id"]))

                    cleanup(opt)
                    pytest.fail()

                if (outer_sort == outer_sort_next):
                    if (inner_sort > inner_sort_next):
                        print("For Halo ID {0} we had a {1} of {2}.  After sorting via lexsort "
                              "inner-key {1}, the next Halo has ID {3} and a {1} of {4}"
                              .format(halo_id, opt["sort_mass"], inner_sort, halo_id_next,
                              inner_sort_next))
                        print("Since we are sorting using {0} they MUST be in ascending "
                              "order.".format(opt["sort_mass"]))

                        cleanup(opt)
                        pytest.fail()
 
def my_test_check_haloIDs(opt):
    """
    
    This test checks the passed haloIDs and snapshot number to ensure the haloIDs match the given formula.
 
    Parameters
    ----------

    opt: Dictionary.  
        Dictionary containing the option parameters specified at runtime.  
        Used to get file name and sorting fields. 
 
    Returns
    ----------

    None. ``Pytest.fail()`` is invoked if the test fails.     

    """
   
    files = [opt["fname_in"], opt["fname_out"]]

    for file_to_test in files: 
        with h5py.File(file_to_test, "r") as f_in:
            Snap_Keys, Snap_Nums = fs.get_snapkeys_and_nums(f_in.keys())
        
            for snap_key in Snap_Keys:
                if len(f_in[snap_key][opt["halo_id"]]) == 0:  # Skip empty snapshots. 
                    continue

                file_haloIDs = f_in[snap_key][opt["halo_id"]][:] 
                generated_haloIDs = fs.index_to_temporalID(np.arange(len(file_haloIDs)), 
                                                                     Snap_Nums[snap_key],
                                                                     opt["index_mult_factor"])
    
        if (np.array_equal(generated_haloIDs, file_haloIDs) == False):
            print("The HaloIDs within file '{0}' were not correct.".format(file_to_test))
            print("HaloIDs were {0} and the expected IDs were {1}.".format(file_haloIDs,
                                                                           generated_haloIDs))
            print("If this is the test input data file, then your input data is wrong!\n"
                  "If this is the test sorted output file, contact jseiler@swin.edu.au.")

            cleanup(opt)
            pytest.fail()

def my_test_sorted_properties(opt):
    """

    Ensures that the halo properties (i.e., the fields that don't contain IDs) were sorted and
    saved properly. 

    Note: The non-ID fields are not checked here because they are wrong by design.  If HaloID 
    1900000000001 had a descendant pointer (i.e., a 'Head' point in Treefrog) of 2100000000003, 
    this may not be true because the ID of Halo 2100000000003 may be changed. 
 
    Parameters
    ----------

    opt: Dictionary.  
        Dictionary containing the option parameters specified at runtime.  
        Used to get file name and sorting fields. 
     
    Returns
    ----------

    True if test passes, False otherwise.

    """
  
    with h5py.File(opt["fname_in"], "r") as f_in, h5py.File(opt["fname_out"], "r") as f_out:
           
        Snap_Keys, Snap_Nums = fs.get_snapkeys_and_nums(f_out.keys())
 
        for snap_key in Snap_Keys:  # Now let's check each field.
            for field in f_out[snap_key]:

                if (field in opt["ID_fields"]) == True:  # If the field is an ID field ignore it (see docstring). 
                    continue

                indices = fs.get_sort_indices(f_in, 
                                           snap_key, opt)  # Gets the indices that would sort the
                                                           # snapshot data. 

                input_data = f_in[snap_key][field][:] # Grab all the data we are going to check.
                input_data_sorted = input_data[indices]
                output_data = f_out[snap_key][field][:]

                if not np.array_equal(output_data, input_data_sorted):
                    print("For snapshot number {0}, there was a mismatch for field {1} between "
                          "the sorted input data and the data stored in the output file."
                          .format(Snap_Nums[snap_key], field))
                    print("The raw input data is {0}.  The supposed indices that would sort this "
                          "data is {1} corresponding to 'sorted' input data of {2}.  However the "
                          "data stored in the output file is {3}".format(input_data, indices,
                          input_data_sorted, output_data))

                    cleanup(opt) 
                    pytest.fail()
    
def create_test_input_data(opt):
    """

    If the user specifies that they will provide a data file, we do not wish to perform test on 
    the entire file.  We first copy a number of halos (Default 10,000) to perform the testing on. 
   
    Note: We copy entire snapshots over meaning that halo counts will not be exact.  If the first
    snapshot with halos has 6,000 halos and the second has 7,000, our testing file will contain
    13,000 halos. 
 
    Parameters
    ----------

    opt: Dictionary.  
        Dictionary containing the option parameters specified at runtime.  
        Used to get file name and number of halos to copy. 
     
    Returns
    ----------

    True if test passes, False otherwise.

    """


    with h5py.File(opt["fname_in"], "r") as f_in, h5py.File("./my_test_data.hdf5", "w") as f_out:
        NHalos = 0
        
        Snap_Keys = [key for key in f_in.keys() if (("SNAP" in key.upper()) == True)] 
        Snap_Nums = dict() 
        for key in Snap_Keys: 
            Snap_Nums[key] = fs.snap_key_to_snapnum(key) 

        for snap_key in Snap_Keys:
            if len(f_in[snap_key][opt["halo_id"]]) == 0:  # Skip empty snapshots. 
                continue
        
            fs.copy_group(f_in, f_out, snap_key, opt)
            NHalos += len(f_in[snap_key][opt["halo_id"]])

            if NHalos >= opt["NHalos_test"]:
                break

    return "./my_test_data.hdf5"

def cleanup(opt):

    if "-f" in sys.argv: # Don't delete the default input data. 
        os.remove(opt["fname_in"])
    os.remove(opt["fname_out"]) 

def test_run():
    """

    Function to run the tests. This will be the only function called by pytest which in turn will
    call all the other test functions.
 
    Parameters
    ----------

    None.
     
    Returns
    ----------

    None 

    """

    opt = parse_inputs()
    opt = vars(opt)  # Cast to dictionary.
    
    if "-f" in sys.argv: # User specified their own input data.
        print("You have supplied your own test input data.")
        print("Saving a small file with the first {0} Halos.".format(opt["NHalos_test"]))
        opt["fname_in"] = create_test_input_data(opt)

    # Since we are generating a sorted file from only a partial number of halos, 
    # the merger pointers could point to a snapshot that is not included.
    # Hence we need to skip all the merger pointer fields.
 
    tmp_ID_fields = opt["ID_fields"]
    opt["ID_fields"] = opt["halo_id"]
    fs.sort_and_write_file(opt)

    opt["ID_fields"] = tmp_ID_fields #  Then put back the old option.

    print("Checking that the produced temporal IDs are correct.")
    my_test_check_haloIDs(opt)  

    print("Checking that the sort order was done/saved correctly for the sort keys.") 
    my_test_sorted_order(opt)  

    print("Checking that the sort order was done/saved correctly for the other halo properties.")
    my_test_sorted_properties(opt)  

    print("")
    print("All tests have passed.")

    cleanup(opt)
     
