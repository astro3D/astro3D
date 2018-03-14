#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from optparse import OptionParser
import sys
import h5py
import os
import pytest

from genesis.utils import forest_sorter as fs
from genesis.utils import common as cmn 

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
                      help="Path to test HDF5 data. Default: "
                      "{0}/test_data.hdf5".format(test_dir), 
                      default="{0}/test_data.hdf5".format(test_dir))
    parser.add_option("-o", "--fname_out", dest="fname_out", 
                      help="Path to sorted output HDF5 data file. Default: "
                       "{0}/test_sorted.hdf5".format(test_dir),
                      default="{0}/test_sorted.hdf5".format(test_dir))
    parser.add_option("-n", "--NHalos_test", dest="NHalos_test", 
                      help="Minimum number of halos to test using. Default: " 
                      "10,000", default = 10000, type = int)
    parser.add_option("-s", "--sort_fields", dest="sort_fields", 
                      help="Field names we will be sorted on. ORDER IS "
                      "IMPORTANT.  Order using the outer-most sort to the "
                      "inner-most. You MUST specify 4 fields to sort "
                      "on (due to the limitations of the optionParser package)"  
                      ".  If you wish to sort on less use None.  If you wish "
                      "to sort on more, email jseiler@swin.edu.au.  Default: "
                      "('ForestID', 'Mass_200mean', None, None)", 
                      default = ("ForestID", "Mass_200mean", None, None),
                      nargs = 4)
    parser.add_option("-i", "--HaloID", dest="halo_id", 
                      help="Field name for halo ID. Default: ID.", 
                      default = "ID")
    parser.add_option("-p", "--ID_fields", dest="ID_fields", 
                      help="Field names for those that contain non-merger."  
                      "  Default: ('ID','Tail', 'Head').", 
                      default = ("ID", "Tail", "Head")) 
    parser.add_option("-x", "--index_mult_factor", dest="index_mult_factor", 
                      help="Conversion factor to go from a unique, "
                      "snapshot-unique halo index temporally unique haloID.  " 
                      "Default: 1e12.", default = 1e12)

    (opt, args) = parser.parse_args()

    # Print some useful startup info. #
    print("")
    print("Running test functions")
    print("Performing tests on a minimum of {0} halos."
          .format(opt.NHalos_test))
    print("The HaloID field for each halo is '{0}'.".format(opt.halo_id))
    print("Sorting on the {0} fields".format(opt.sort_fields))
    print("")

    return opt


def recursively_check_sort(snapshot_data, opt, sort_level, halo_idx):
    """
    Moves through the sort level, checking that each key was sorted.

    Parameters
    ----------

    snapshot_data: HDF5 File. Required. 
        Snapshot data that we are checking.  The fields of this are the halo
        properties for the snapshot. 

    opt: Dictionary. Required.
        Dictionary containing the option parameters specified at runtime.  
        Used to get the sorting fields. 

    sort_level: Integer. Required.
        The sort level that we are currently on.  Used to get the sort key.

    halo_idx: Integer. Required.
        Index of the halo we are comparing.

    Returns
    ----------

    None. ``Pytest.fail()`` is invoked if the test fails. 
    """

    # Our checking goes from outer-most to inner-most.  If the user didn't want
    # to sort on 4 fields and used None, then we stop recursively calling. 
    key = opt["sort_fields"][sort_level]
    if key is None or "NONE" in key.upper():        
        return 

    this_value = snapshot_data[key][halo_idx]
    this_id = snapshot_data[key][halo_idx]

    next_value = snapshot_data[key][halo_idx + 1]
    next_id = snapshot_data[key][halo_idx]

    # If the values are equal, we need to move to the next sort level.  However
    # if we're currently at the inner-most level then the sorting is still done
    # correctly (equal values next to each other). 
    if this_value == next_value \
       and sort_level < (len(opt["sort_fields"]) - 1): 
        recursively_check_sort(snapshot_data, opt, sort_level + 1,
                               halo_idx)

    # Otherwise if we haven't sorted correctly in ascended order, print a
    # message and fail the test.        
    elif this_value > next_value:
        print("For Halo ID {0} we had a {1} value of {2}.  After sorting "
              "via lexsort using the fields {3} (inner-most sort first), " 
              "the next in the sorted list has ID {4} and a {1} value of {5}"
              .format(this_id, key, this_id, opt["sort_fields"],
                      next_id, key, next_id))

        cleanup(opt)
        pytest.fail() 
               
    return 

def my_test_sorted_order(opt):
    """
    Checks the indices of the output file to ensure sorting order is correct.

    Calls ``recursively_check_sort`` for each halo which iterates through the
    sorted fields to ensure all the sorted is correct.

    Parameters
    ----------

    opt: Dictionary. Required. 
        Dictionary containing the option parameters specified at runtime.  
        Used to get file name and sorting fields. 

    Returns
    ----------

    None. ``Pytest.fail()`` is invoked by ``recursively_check_sort`` if the
    test fails.
    """

    with h5py.File(opt["fname_out"], "r") as f_in:
           
        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        for snap_key in Snap_Keys:
            NHalos = len(f_in[snap_key][opt["halo_id"]])
            if NHalos < 2:  # Skip snapshots that wouldn't be sorted. 
                continue
 
            # Since the user specifies 4 keys that they wish to sort on (with
            # some these potentially being None), we need to check that every
            # key has been sorted correctly.
            #
            # To do this we loop over the halos within a snapshot and first
            # check the outer-most key.  If halo[i] has the same outer-key as
            # halo[i + 1] we need to check an inner-key to ensure it's sorted. 

            for idx in range(NHalos - 1):
                recursively_check_sort(f_in[snap_key], opt, 0, idx)
 
def my_test_check_haloIDs(opt):
    """
    Checks the sorted haloIDs and snapshot numbers match the formula.           

    This formula is the one that turns the snapshot-local halo index into a
    temporally unique ID.
 
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
            Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())
        
            for snap_key in Snap_Keys:
                if len(f_in[snap_key][opt["halo_id"]]) == 0:  # Skip empty snapshots. 
                    continue

                file_haloIDs = f_in[snap_key][opt["halo_id"]][:] 
                generated_haloIDs = cmn.index_to_temporalID(np.arange(len(file_haloIDs)), 
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
           
        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_out.keys())
 
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

    If the user asks to test on more halos than there are in the data file we raise a RuntimeError.
 
    Parameters
    ----------

    opt: Dictionary.  
        Dictionary containing the option parameters specified at runtime.  
        Used to get file name and number of halos to copy. 
     
    Returns
    ----------

    fname_out: String.
        The path to the small copied data file. 

    """

    test_dir = os.path.dirname(__file__)
    fname_out = "{0}/my_test_data.hdf5"

    with h5py.File(opt["fname_in"], "r") as f_in, h5py.File(fname_out, "w") as f_out:
        NHalos = 0
        
        Snap_Keys = [key for key in f_in.keys() if (("SNAP" in key.upper()) == True)] 
        Snap_Nums = dict() 
        for key in Snap_Keys: 
            Snap_Nums[key] = cmn.snap_key_to_snapnum(key) 

        for snap_key in Snap_Keys:
            if len(f_in[snap_key][opt["halo_id"]]) == 0:  # Skip empty snapshots. 
                continue
        
            cmn.copy_group(f_in, f_out, snap_key, opt)
            NHalos += len(f_in[snap_key][opt["halo_id"]])

            if NHalos >= opt["NHalos_test"]:
                break

    if NHalos < opt["NHalos_test"]:
        print("Your supplied data file did not contain enough halos to test on.")
        print("Your file contained {0} whereas you specified to run on {1} halos."
              .format(NHalos, opt["NHalos_test"]))
        print("Either lower the number of halos to test on (--Nhalos_test) or use other data.")
        raise RuntimeError

    return fname_out 

def cleanup(opt):
    """

    Remove the output sorted test data and if the user specified their own data to use, remove the
    small chunk of data we created. 
 
    Parameters
    ----------

    opt: Dictionary.  
        Dictionary containing the option parameters specified at runtime.  
        Used to get file names. 
         
    Returns
    ----------

    None 

    """

    if "-f" in sys.argv: # Don't delete the default input data. 
        os.remove(opt["fname_in"])
    os.remove(opt["fname_out"]) 

def test_run():
    """

    Function to run the tests. This will be the only function called by pytest which in turn will
    call all the other test functions.

    Alternatively if the user wishes to specify their own input data, this function will be called
    by main.
 
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
    

if __name__ == "__main__":

    test_run()
