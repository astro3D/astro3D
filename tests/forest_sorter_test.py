#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import argparse
import sys
import h5py
import os
import pytest

from genesis.utils import forest_sorter as fs
from genesis.utils import common as cmn


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
    test_dir = os.path.dirname(__file__)

    parser.add_argument("-f", "--fname_in", dest="fname_in",
                        help="Path to test HDF5 data. Default: "
                        "{0}/test_data.hdf5".format(test_dir),
                        default="{0}/test_data.hdf5".format(test_dir))
    parser.add_argument("-o", "--fname_out", dest="fname_out",
                        help="Path to sorted output HDF5 data file. "
                        "Default: {0}/test_sorted.hdf5".format(test_dir),
                        default="{0}/test_sorted.hdf5".format(test_dir))
    parser.add_argument("-s", "--sort_fields", dest="sort_fields",
                        help="Field names we will be sorted on. ORDER IS "
                        "IMPORTANT.  Order using the outer-most sort to the "
                        "inner-most.  Separate each field name with a comma. "
                        "Default: ForestID,Mass_200mean.",
                        default="ForestID,Mass_200mean")
    parser.add_argument("-i", "--HaloID", dest="halo_id",
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

    args = parser.parse_args()

    # We allow the user to enter an arbitrary number of sort fields and fields
    # that contain IDs.  They are etnered as a single string separated by
    # commas so need to split them up into a list.
    args.ID_fields = (args.ID_fields).split(',')
    args.sort_fields = args.sort_fields.split(',')

    # Print some useful startup info. #
    print("")
    print("Running test functions")
    print("Performing tests on a minimum of {0} halos."
          .format(args.NHalos_test))
    print("The HaloID field for each halo is '{0}'.".format(args.halo_id))
    print("Sorting on the {0} fields".format(args.sort_fields))
    print("")

    return vars(args)


def recursively_check_sort(snapshot_data, args, sort_level, halo_idx):
    """
    Moves through the sort level, checking that each key was sorted.

    Parameters
    ----------

    snapshot_data: HDF5 File. Required.
        Snapshot data that we are checking.  The fields of this are the halo
        properties for the snapshot.

    args: Dictionary. Required.
        Dictionary containing the argsion parameters specified at runtime.
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
    key = args["sort_fields"][sort_level]
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
       and sort_level < (len(args["sort_fields"]) - 1):
        recursively_check_sort(snapshot_data, args, sort_level + 1,
                               halo_idx)

    # Otherwise if we haven't sorted correctly in ascended order, print a
    # message and fail the test.
    elif this_value > next_value:
        print("For Halo ID {0} we had a {1} value of {2}.  After sorting "
              "via lexsort using the fields {3} (inner-most sort first), "
              "the next in the sorted list has ID {4} and a {1} value of {5}"
              .format(this_id, key, this_id, args["sort_fields"],
                      next_id, key, next_id))

        cleanup(args)
        pytest.fail()

    return


def my_test_sorted_order(args):
    """
    Checks the indices of the output file to ensure sorting order is correct.

    Calls ``recursively_check_sort`` for each halo which iterates through the
    sorted fields to ensure all the sorted is correct.

    Parameters
    ----------

    args: Dictionary. Required.
        Dictionary containing the argsion parameters specified at runtime.
        Used to get file name and sorting fields.

    Returns
    ----------

    None. ``Pytest.fail()`` is invoked by ``recursively_check_sort`` if the
    test fails.
    """

    with h5py.File(args["fname_out"], "r") as f_in:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        for snap_key in Snap_Keys:
            NHalos = len(f_in[snap_key][args["halo_id"]])
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
                recursively_check_sort(f_in[snap_key], args, 0, idx)


def my_test_check_haloIDs(args):
    """
    Checks the sorted haloIDs and snapshot numbers match the formula.

    This formula is the one that turns the snapshot-local halo index into a
    temporally unique ID.

    Parameters
    ----------

    args: Dictionary.
        Dictionary containing the argsion parameters specified at runtime.
        Used to get file name and sorting fields.

    Returns
    ----------

    None. ``Pytest.fail()`` is invoked if the test fails.
    """

    files = [args["fname_in"], args["fname_out"]]

    for file_to_test in files:
        with h5py.File(file_to_test, "r") as f_in:
            Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

            for snap_key in Snap_Keys:
                if len(f_in[snap_key][args["halo_id"]]) == 0:
                    continue

                file_haloIDs = f_in[snap_key][args["halo_id"]][:]
                generated_haloIDs = cmn.index_to_temporalID(np.arange(len(file_haloIDs)),
                                                            Snap_Nums[snap_key],
                                                            args["index_mult_factor"])

        if not np.array_equal(generated_haloIDs, file_haloIDs):
            print("The HaloIDs within file '{0}' were not correct."
                  .format(file_to_test))
            print("HaloIDs were {0} and the expected IDs were {1}."
                  .format(file_haloIDs, generated_haloIDs))
            print("If this is the test input data file, then your input data "
                  "may be wrong!  If this is the test sorted output file, "
                  "contact jseiler@swin.edu.au")

            cleanup(args)
            pytest.fail()


def my_test_sorted_properties(args):
    """
    Ensures that the halo properties were sorted and saved properly.

    Note: The non-ID fields are not checked here because they are
    wrong by design.  If HaloID 1900000000001 had a descendant pointer
    (i.e., a 'Head' point in Treefrog) of 2100000000003, this may not
    be true because the ID of Halo 2100000000003 may be changed.

    Parameters
    ----------

    args: Dictionary.
        Dictionary containing the argsion parameters specified at runtime.
        Used to get file name and sorting fields.

    Returns
    ----------

    True if test passes, False otherwise.
    """

    with h5py.File(args["fname_in"], "r") as f_in, \
         h5py.File(args["fname_out"], "r") as f_out:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_out.keys())

        for snap_key in Snap_Keys:  # Now let's check each field.
            for field in f_out[snap_key]:

                if field in args["ID_fields"]:  # Ignore ID fields.
                    continue

                indices = fs.get_sort_indices(f_in,
                                              snap_key, args)

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

                    cleanup(args)
                    pytest.fail()


def create_test_input_data(args):
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

    args: Dictionary.
        Dictionary containing the argsion parameters specified at runtime.
        Used to get file name and number of halos to copy.

    Returns
    ----------

    fname_out: String.
        The path to the small copied data file.
    """

    fname_out = "{0}/my_test_data.hdf5"

    with h5py.File(args["fname_in"], "r") as f_in, \
         h5py.File(fname_out, "w") as f_out:
        NHalos = 0

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        for snap_key in Snap_Keys:
            if len(f_in[snap_key][args["halo_id"]]) == 0:
                continue

            cmn.copy_group(f_in, f_out, snap_key, args)
            NHalos += len(f_in[snap_key][args["halo_id"]])

            if NHalos >= args["NHalos_test"]:
                break

    if NHalos < args["NHalos_test"]:
        print("Your supplied data file did not contain enough halos to test.")
        print("Your file contained {0} halos whereas you specified to run "
              "on {1} halos.".format(NHalos, args["NHalos_test"]))
        print("Either lower the number of halos to test on (--Nhalos_test) or "
              "use other data.")
        raise RuntimeError

    return fname_out


def cleanup(args):
    """
    Remove the output sorted test data.

    If the user specified their own data to test, remove the small chunk we
    copied.

    Parameters
    ----------

    args: Dictionary.
        Dictionary containing the argsion parameters specified at runtime.
        Used to get file names.

    Returns
    ----------

    None
    """

    if "-f" in sys.argv:  # Don't delete the default input data.
        os.remove(args["fname_in"])
    os.remove(args["fname_out"])


def test_run():
    """
    Wrapper to run all the tests.

    Parameters
    ----------

    None.

    Returns
    ----------

    None.
    """

    args = parse_inputs()

    if "-f" in sys.argv:  # User specified their own input data.
        print("You have supplied your own test input data.")
        print("Saving a small file with the first {0} Halos."
              .format(args["NHalos_test"]))
        args["fname_in"] = create_test_input_data(args)

    # Since we are generating a sorted file from only a partial number of halos
    # the merger pointers could point to a snapshot that is not included.
    # Hence we need to skip all the merger pointer fields.

    tmp_ID_fields = args["ID_fields"]
    args["ID_fields"] = args["halo_id"]
    fs.sort_and_write_file(args)

    args["ID_fields"] = tmp_ID_fields  # Then put back the old argsion.

    print("Checking that the produced temporal IDs are correct.")
    my_test_check_haloIDs(args)

    print("Checking that the sort order is correct for the sort keys.")
    my_test_sorted_order(args)

    print("Checking that the sort order is correct for the halo properties.")
    my_test_sorted_properties(args)

    print("")
    print("All tests have passed.")

    cleanup(args)


if __name__ == "__main__":

    test_run()
