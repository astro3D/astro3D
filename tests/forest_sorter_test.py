#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import argparse
import sys
import h5py
import os
import pytest
from tqdm import tqdm

from genesis.utils import forest_sorter as fs
from genesis.utils import common as cmn


test_dir = os.path.dirname(os.path.realpath(__file__))
default_fname_in = "{0}/test_data.hdf5".format(test_dir)
default_user_fname_in = "{0}/my_test_data.hdf5".format(test_dir)
default_fname_out = "{0}/test_sorted.hdf5".format(test_dir)

def parse_inputs():
    """
    Parses the command line input arguments.

    If there has not been an input or output file specified a RuntimeError will
    be raised.

    Parameters
    ----------

    None.

    Returns
    ----------

    args: Dictionary.  Required.
        Dictionary of arguments from the ``argparse`` package.
        Dictionary is keyed by the argument name (e.g., args['fname_in']).
    """

    parser = argparse.ArgumentParser()


    parser.add_argument("-f", "--fname_in", dest="fname_in",
                        help="Path to test HDF5 data. Default: "
                        "{0}".format(default_fname_in),
                        default=default_fname_in)
    parser.add_argument("-o", "--fname_out", dest="fname_out",
                        help="Path to sorted output HDF5 data file. "
                        "Default: {0}".format(default_fname_out), 
                        default=default_fname_out)
    parser.add_argument("-s", "--sort_fields", dest="sort_fields",
                        help="Field names we will be sorted on. ORDER IS "
                        "IMPORTANT.  Order using the outer-most sort to the "
                        "inner-most.  Separate each field name with a comma. "
                        "Default: ForestID,Mass_200mean.",
                        default="ForestID,Mass_200mean")
    parser.add_argument("-i", "--HaloID", dest="haloID_field",
                        help="Field name for halo ID. Default: ID.",
                        default="ID")
    parser.add_argument("-p", "--ID_fields", dest="ID_fields",
                        help="Field names for those that contain IDs.  Separate "
                        "field names with a comma. "
                        "Default: ID,Tail,Head,NextSubHalo,Dummy1,Dumm2).",
                        default=("ID,Tail,Head,NextSubHalo,Dummy,Dummy"))
    parser.add_argument("-x", "--index_mult_factor", dest="index_mult_factor",
                        help="Conversion factor to go from a unique, "
                        "per-snapshot halo index to a temporally unique haloID. "
                        "Default: 1e12.", default=1e12)
    parser.add_argument("-n", "--NHalos_test", dest="NHalos_test",
                        help="Minimum number of halos to test. Default: "
                        "10,000", default=10000, type=int)
    parser.add_argument("-g", "--gen_data", dest="gen_data",
                        help="Flag whether we want to generate data. If this " 
                             "is set to 0, the tests will be run on the " 
                             "`fname_out` sorted data that was created "
                             "running on `fname_in`. Default: True.",
                             default=1, type=int)

    args = parser.parse_args()

    # We allow the user to enter an arbitrary number of sort fields and fields
    # that contain IDs.  They are etnered as a single string separated by
    # commas so need to split them up into a list.
    args.ID_fields = (args.ID_fields).split(',')
    args.sort_fields = args.sort_fields.split(',')
   
    if not args.gen_data and (args.fname_in == default_fname_in or
                              args.fname_out == default_fname_out):
        print("You specified that you do not want to generate data and instead "
              "want to test an already sorted HDF5 file.")
        print("For this setting, you must specify the ORIGINAL UNSORTED HDF5 "
              "trees using the --fname_in option and the SORTED HDF5 trees "
              "using the --fname_out option.")
        raise ValueError 

    # Print some useful startup info. #
    print("")
    print("Running test functions")
    print("Performing tests on a minimum of {0} halos."
          .format(args.NHalos_test))
    print("The HaloID field for each halo is '{0}'.".format(args.haloID_field))
    print("Sorting on the {0} fields".format(args.sort_fields))
    print("")

    return vars(args)


def my_test_sorted_order(fname_out=default_fname_out, haloID_field="ID", 
                         sort_fields=["ForestID", "hostHaloID", "Mass_200mean"], 
                         gen_data=0):
    """
    Checks the indices of the output file to ensure sorting order is correct.

    Calls `recursively_check_sort` for each halo which iterates through the
    sorted fields to ensure all the sorted is correct.

    Parameters
    ----------

    fname_out: String. Default: `<test_directory>/test_sorted.hdf5`. 
        Path to the sorted HDF5 trees we're testing. 
        saved.

        ..note::
            If `gen_data=1` the sorted trees will be removed upon exit. 

    haloID_field: String. Default: 'ID'.
        Field name within the HDF5 file that corresponds to the unique halo ID.

    sort_fields: List of strings. Default: ['ForestID', 'hostHaloID',
                                            'Mass_200mean'].
        The HDF5 field names that the sorting was performed on. The entries
        are ordered such that the first field will be the outer-most sort and
        the last field will be the inner-most sort.

    gen_data: Integer. Default: 0.
        Flag whether this function was called using sorted trees generated for
        testing purposes.  If this flag is set to 1 then the sorted trees will
        be removed if the test fails. 

    Returns
    ----------

    None. 

    `~pytest.fail()` is invoked by `recursively_check_sort()` if the test fails.
    """

    def recursively_check_sort(snapshot_data, sort_fields, sort_level, 
                               halo_idx, gen_data):
        """
        Moves through the sort level, checking that each key was sorted.
        """

        # Our checking goes from outer-most to inner-most.  If the user didn't want
        # to sort on 4 fields and used None, then we stop recursively calling.
        key = sort_fields[sort_level]
        if key is None or "NONE" in key.upper():
            return

        values = snapshot_data[key][:]

        this_value = values[halo_idx]
        this_id = values[halo_idx]

        next_value = values[halo_idx + 1]
        next_id = values[halo_idx + 1]

        # If the values are equal, we need to move to the next sort level.  However
        # if we're currently at the inner-most level then the sorting is still done
        # correctly (equal values next to each other).
        if this_value == next_value \
           and sort_level < (len(sort_fields) - 1):
            recursively_check_sort(snapshot_data, sort_fields,
                                   sort_level + 1, halo_idx, gen_data)

        # Otherwise if we haven't sorted correctly in ascended order, print a
        # message and fail the test.
        elif this_value > next_value:
            print("For Halo ID {0} we had a {1} value of {2}.  After sorting "
                  "via lexsort using the fields {3} (inner-most sort first), "
                  "the next in the sorted list has ID {4} and a {1} value of {5}"
                  .format(this_id, key, this_id, sort_fields,
                          next_id, next_id))

            if gen_data:
                cleanup()
            pytest.fail()

        return None

    with h5py.File(fname_out, "r") as f_in:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        print("Looping over each snapshot.")
        for snap_key in tqdm(Snap_Keys):
            NHalos = len(f_in[snap_key][haloID_field])
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
                recursively_check_sort(f_in[snap_key], sort_fields, 0, 
                                       idx, gen_data)


def my_test_check_haloIDs(fname_in=default_fname_in,
                          fname_out=default_fname_out, haloID_field="ID", 
                          index_mult_factor=1e12, gen_data=0):
    """
    Checks the sorted haloIDs and snapshot numbers match the formula.

    This formula is the one that turns the snapshot-local halo index into a
    temporally unique ID.

    Parameters
    ----------

    fname_in, fname_out: String. Default: `<test_directory>/test_data.hdf5`,
    `<test_directory>/test_sorted.hdf5`.     
        Path to the input HDF5 trees and path to where the sorted trees were
        saved. 

    haloID_field: String. Default: 'ID'.
        Field name within the HDF5 file that corresponds to the unique halo ID.

    index_mult_factor: Integer. Default: 1e12.
        Multiplication factor to generate a temporally unique halo ID. See
        `common.index_to_temporalID`.

    gen_data: Integer. Default: 0.
        Flag whether this function was called using sorted trees generated for
        testing purposes.  If this flag is set to 1 then the sorted trees will
        be removed if the test fails. 

    Returns
    ----------

    None. 

    `~pytest.fail()` is invoked if the test fails. 
    """

    files = [fname_in, fname_out]

    for file_to_test in files:
        print("Checking that the HaloIDs are correct for file "
              "{0}".format(file_to_test))
        with h5py.File(file_to_test, "r") as f_in:
            Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

            print("Looping over each Snapshot.")
            for snap_key in tqdm(Snap_Keys):
                if len(f_in[snap_key][haloID_field]) == 0:
                    continue

                file_haloIDs = f_in[snap_key][haloID_field][:]
                generated_haloIDs = cmn.index_to_temporalID(np.arange(len(file_haloIDs)),
                                                            Snap_Nums[snap_key],
                                                            index_mult_factor)

        if not np.array_equal(generated_haloIDs, file_haloIDs):
            print("The HaloIDs within file '{0}' were not correct."
                  .format(file_to_test))
            print("HaloIDs were {0} and the expected IDs were {1}."
                  .format(file_haloIDs, generated_haloIDs))
            print("If this is the test input data file, then your input data "
                  "may be wrong!  If this is the test sorted output file, "
                  "contact jseiler@swin.edu.au")

            if gen_data:
                cleanup(fname_in)
            pytest.fail()


def my_test_sorted_properties(fname_in=default_fname_in,
                              fname_out=default_fname_out, ID_fields="ID", 
                              sort_fields=["ForestID", "hostHaloID",
                                           "Mass_200mean"], gen_data=0): 
    """
    Ensures that the halo properties were sorted and saved properly.

    Note: The non-ID fields are not checked here because they are
    wrong by design.  If HaloID 1900000000001 had a descendant pointer
    (i.e., a 'Head' point in Treefrog) of 2100000000003, this may not
    be true because the ID of Halo 2100000000003 may be changed.

    Parameters
    ----------

    fname_in, fname_out: String. Default: `<test_directory>/test_data.hdf5`,
    `<test_directory>/test_sorted.hdf5`.     
        Path to the input HDF5 trees and path to where the sorted trees were
        saved. 

    haloID_field: String. Default: 'ID'.
        Field name within the HDF5 file that corresponds to the unique halo ID.

    sort_fields: List of strings. Default: ['ForestID', 'hostHaloID',
                                            'Mass_200mean'].
        The HDF5 field names that the sorting was performed on. The entries
        are ordered such that the first field will be the outer-most sort and
        the last field will be the inner-most sort.

    gen_data: Integer. Default: 0.
        Flag whether this function was called using sorted trees generated for
        testing purposes.  If this flag is set to 1 then the sorted trees will
        be removed if the test fails. 

    Returns
    ----------

    None. ``Pytest.fail()`` is invoked if the test fails.
    """

    with h5py.File(fname_in, "r") as f_in, \
         h5py.File(fname_out, "r") as f_out:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_out.keys())

        print("Looping over each Snapshot")
        for snap_key in tqdm(Snap_Keys):  # Now let's check each field.
            for field in f_out[snap_key]:

                if field in ID_fields:  # Ignore ID fields.
                    continue

                indices = fs.get_sort_indices(f_in,
                                              snap_key, sort_fields)

                input_data = f_in[snap_key][field][:]
                input_data_sorted = input_data[indices]
                output_data = f_out[snap_key][field][:]

                if not np.array_equal(output_data, input_data_sorted):
                    print("For snapshot number {0}, there was a mistmach for "
                          "field {1} between the sorted input data and the "
                          "data stored in the output file."
                          .format(Snap_Nums[snap_key], field))
                    print("The raw input data is {0}.  The supposed indices "
                          "that would sort this data is {1} corresponding to "
                          "'sorted' input data of {2}.  However the data "
                          "stored in the output file is {3}"
                          .format(input_data, indices, input_data_sorted,
                                  output_data))

                    if gen_data:
                        cleanup(fname_in)
                    pytest.fail()


def create_test_input_data(fname_in=default_fname_in, haloID_field="ID", 
                           NHalos_test=10000):
    """
    Creates a test data set from the user supplied input data.

    Copies over a specified number of halos (Default 10,000) to perform the
    testin on.

    Note: We copy entire snapshots over meaning that halo counts will not be
    exact.  If the first snapshot with halos has 6,000 halos and the second
    has 7,000, our testing file will contain 13,000 halos.

    If the user asks to test on more halos than there are in the data file
    we raise a RuntimeError.

    Parameters
    ----------

    fname_in: String. Default: `<test_directory>/test_data.hdf5`.
        Path to the HDF5 trees we're creating the data set from.

    haloID_field: String. Default: 'ID'.
        Field name within the HDF5 file that corresponds to the unique halo ID.

    NHalos_test: Integer. Default: 10000.
        The number of halos that will be copied into the test data set.

    Returns
    ----------

    default_user_fname_in: String.
        The path to the small copied data file.
    """

    with h5py.File(fname_in, "r") as f_in, \
         h5py.File(default_user_fname_in, "w") as f_out:
        NHalos = 0

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        for snap_key in Snap_Keys:
            if len(f_in[snap_key][haloID_field]) == 0:
                continue

            cmn.copy_group(f_in, f_out, snap_key)
            NHalos += len(f_in[snap_key][haloID_field])

            if NHalos >= NHalos_test:
                break

    if NHalos < NHalos_test:
        print("Your supplied data file did not contain enough halos to test.")
        print("Your file contained {0} halos whereas you specified to run "
              "on {1} halos.".format(NHalos, NHalos_test))
        print("Either lower the number of halos to test on (--Nhalos_test) or "
              "use other data.")
        raise RuntimeError

    return default_user_fname_in 


def cleanup(fname_in):
    """
    If the user generated a small chunk of data to test on, delete it and the
    resulting sorted trees. 

    ..note::
        This function is only called if `gen_data=1` is passed to the test
        functions.

    Parameters
    ----------

    None.

    Returns
    ----------

    None.
    """

    # Check to see if a small test data set was created.    
    if os.path.isfile(default_user_fname_in):
        os.remove(fname_in)

    # Remove the sorted trees. 
    os.remove(default_fname_out)


def test_run(fname_in=default_fname_in, fname_out=default_fname_out, 
             haloID_field="ID", sort_fields=["ForestID", "hostHaloID", 
                                             "Mass_200mean"],
             ID_fields=["Head", "Tail", "RootHead", "RootTail", 
                        "ID", "hostHaloID"], 
             index_mult_factor=1e12, NHalos_test=10000, gen_data=1):
    """
    Wrapper to run all the tests.

    Default parameters are set for ASTRO3D Genesis trees. 

    Parameters
    ----------

    fname_in, fname_out: String. Default: `<test_directory>/test_data.hdf5`,
    `<test_directory>/test_sorted.hdf5`. 
        Path to the input HDF5 trees and path to where the sorted trees will be
        saved.

        ..note::
            If `gen_data=1` the sorted trees will be removed upon exit. 

    haloID_field: String. Default: 'ID'.
        Field name within the HDF5 file that corresponds to the unique halo ID.

    sort_fields: List of strings. Default: ['ForestID', 'hostHaloID',
                                            'Mass_200mean'].
        The HDF5 field names that the sorting will be performed on. The entries
        are ordered such that the first field will be the outer-most sort and
        the last field will be the inner-most sort.

    ID_fields: List of string. Default: ['Head', 'Tail', 'RootHead', 'RootTail',
                                        'ID', 'hostHaloID'].
        The HDF5 field names that correspond to properties that use halo IDs.
        As the halo IDs are updated to reflect the new sort order, these fields
        must also be updated. 

    index_mult_factor: Integer. Default: 1e12.
        Multiplication factor to generate a temporally unique halo ID. See
        `common.index_to_temporalID`.

    NHalos_test: Integer. Default: 10000.
        If `gen_data=1`, we only test on a subset of the input HDF5 trees.
        This parameter controls how many halos we use from the input HDF5 trees
        to test on.

    gen_data: Integer. Default: 1.
        Flag whether we want to generate data to test on. If this  is set to 
        0, the tests will be run on the `fname_out` sorted data that was 
        created running on the `fname_in` HDF5 trees. 

    Returns
    ----------

    None.
    """

    if gen_data: 
        if fname_in != default_fname_in:  # User specified their own input data.
            print("You have supplied your own test input data.")
            print("Saving a small file with the first {0} Halos."
                  .format(NHalos_test))
            fname_in = create_test_input_data(fname_in, haloID_field,
                                              NHalos_test)

        # Since we are generating a sorted file from only a partial number of halos
        # the merger pointers could point to a snapshot that is not included.
        # Hence we need to skip all the merger pointer fields.
        tmp_ID_fields = ID_fields
        ID_fields = haloID_field

        fs.forest_sorter(fname_in, fname_out, haloID_field,
                         sort_fields, ID_fields, index_mult_factor)
                         
        ID_fields = tmp_ID_fields  # Then put back the old argsion.

    print("=================================")
    print("Running tests")
    print("Input Unsorted Trees: {0}".format(fname_in))
    print("Output Sorted Trees: {0}".format(fname_out))
    print("Sort Fields: {0}".format(sort_fields))
    print("=================================")
    print("")


    print("Checking that the produced temporal IDs are correct.")
    my_test_check_haloIDs(fname_in, fname_out, haloID_field, 
                          index_mult_factor, gen_data)

    print("Checking that the sort order is correct for the sort keys.")

    my_test_sorted_order(fname_out, haloID_field, 
                         sort_fields, gen_data)

    print("Checking that the sort order is correct for the halo properties.")
    my_test_sorted_properties(fname_in, fname_out,
                              haloID_field, sort_fields)

    print("")
    print("=================================")
    print("All tests passed!")
    print("=================================")

    if gen_data:
        cleanup(fname_in)


if __name__ == "__main__":
    args = parse_inputs()

    test_run(args["fname_in"], args["fname_out"], args["haloID_field"],
             args["sort_fields"], args["ID_fields"], args["index_mult_factor"],
             args["NHalos_test"], args["gen_data"])
