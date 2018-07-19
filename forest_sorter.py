#!/usr/bin:env python
from __future__ import print_function
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import time

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

    parser.add_argument("-f", "--fname_in", dest="fname_in",
                        help="Path to the input HDF5 data file. Required.")
    parser.add_argument("-o", "--fname_out", dest="fname_out",
                        help="Path to the output HDF5 data file. Required.")
    parser.add_argument("-s", "--sort_fields", dest="sort_fields",
                        help="Field names we will be sorted on. ORDER IS "
                        "IMPORTANT.  Order using the outer-most sort to the "
                        "inner-most.  Separate each field name with a comma. "
                        "Default: ForestID,hostHaloID,Mass_200mean.",
                        default="ForestID,hostHaloID,Mass_200mean")
    parser.add_argument("-i", "--HaloID", dest="halo_id",
                        help="Field name for halo ID. Default: ID.",
                        default="ID")
    parser.add_argument("-p", "--ID_fields", dest="ID_fields",
                        help="Field names for those that contain IDs.  "
                        "Separate field names with a comma. "
                        "Default: Head,Tail,RootHead,RootTail,ID,hostHaloID", 
                        default=("Head,Tail,RootHead,RootTail,ID,hostHaloID"))
    parser.add_argument("-x", "--index_mult_factor", dest="index_mult_factor",
                        help="Conversion factor to go from a unique, "
                        "per-snapshot halo index to a temporally unique haloID. "
                        "Default: 1e12.", default=1e12)

    args = parser.parse_args()

    # We require an input file and an output one.
    if (args.fname_in is None or args.fname_out is None):
        parser.print_help()
        raise RuntimeError

    # We allow the user to enter an arbitrary number of sort fields and fields
    # that contain IDs.  They are etnered as a single string separated by
    # commas so need to split them up into a list.
    args.ID_fields = (args.ID_fields).split(',')
    args.sort_fields = args.sort_fields.split(',')

    # Print some useful startup info. #
    print("")
    print("The HaloID field for each halo is '{0}'.".format(args.halo_id))
    print("Sorting on the {0} fields".format(args.sort_fields))
    print("The fields that contain IDs are {0}".format(args.ID_fields))
    print("")

    return vars(args)


def get_sort_indices(file_in, snap_key, args):
    """
    Gets the indices that will sort the HDF5 file.

    This sorting uses the fields provided by the user in args. The sort fields
    (or sort keys) we ordered such that the first key will peform the
    outer-most sort and the last key will perform the inner-most sort.

    Example:
        args["sort_fields"] = ("ForestID", "Mass_200mean")
        ForestID = [1, 4, 39, 1, 1, 4]
        Mass_200mean = [4e9, 10e10, 8e8, 7e9, 3e11, 5e6]

        Then the indices would be [0, 3, 4, 5, 1, 2]

    Parameters
    ----------

    file_in: HDF5 file.  Required.
        Open HDF5 file that we are sorting for. The data structure is assumed
        to be HDF5_File -> Snapshot_Keys -> Halo properties.

    snap_key: String.  Required.
        The field name for the snapshot we are accessing.

    args: Dictionary.  Required.
        Dictionary containing the argsion parameters specified at runtime.
        Used to specify the field names we are sorting on.

    Returns
    ----------

    indices: numpy-array.  Required.
        Array containing the indices that sorts the data using the specified
        sort keys.
    """

    sort_keys = []
    for key in reversed(args["sort_fields"]):
        if key is None or "NONE" in key.upper():
            continue
        sort_keys.append(file_in[snap_key][key])

    indices = np.lexsort((sort_keys))

    return indices


def sort_and_write_file(args):
    """
    Using the argsions specified by the command line, sorts the HDF5
    file by the specified ID field and then sub-sorts by the specified
    mass field.

    The output file will be saved in this sorted order.

    Parameters
    ----------

    args: Dictionary.  Required.
        Contains the runtime variables such as input/output file names
        and fields required for sorting.
        For full contents of the dictionary refer to ``parse_inputs``.

    Returns
    ----------

    None.
    """

    with h5py.File(args["fname_in"], "r") as f_in, \
         h5py.File(args["fname_out"], "w") as f_out:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        ID_maps = dict()
        snapshot_indices = dict()

        print("")
        print("Generating the dictionary to map the oldIDs to the newIDs.")

        start_time = time.time()
        for snap_key in tqdm(Snap_Keys):
            # We only want to go through snapshots that contain halos.
            if len(f_in[snap_key][args["halo_id"]]) == 0:
                continue

            # Need to get the indices that sort the data according to the
            # specified keys.

            indices = get_sort_indices(f_in, snap_key, args)

            old_haloIDs = f_in[snap_key][args["halo_id"]][:]
            old_haloIDs_sorted = old_haloIDs[indices]

            # The ID of a halo depends on its snapshot-local index.
            # As the new haloIDs will be sorted correctly, their index will
            # simply be np.arange(len(Number of Halos)).
            new_haloIDs = cmn.index_to_temporalID(np.arange(len(indices)),
                                                  Snap_Nums[snap_key],
                                                  args["index_mult_factor"])

            oldIDs_to_newIDs = dict(zip(old_haloIDs_sorted, new_haloIDs))

            # Now we've generated the Dicts for this snap, put them into the
            # global dictionary.  We key the ID Dict by the snapshot number
            # rather than the field name because we can access the snapshot
            # number using the oldID.

            snapshot_indices[snap_key] = indices
            ID_maps[Snap_Nums[snap_key]] = oldIDs_to_newIDs

        # For some ID fields (e.g., NextProgenitor), the value is -1. 
        # When we convert from the temporalID to a snapshot number, we subtract
        # 1 and divide by the multiplication factor (default 1e12), then cast
        # to an integer. Hence -2 divided by a huge number will be less than 1
        # and when it's cast to an integer will result in 0.
        # So the 'Snapshot Number' for values of -1 will be 0.  We want to
        # preserve these -1 flags so we map -1 to -1.
        ID_maps[0] = {-1:-1}

        end_time = time.time()
        print("Creation of dictionary map took {0:3f} seconds"
              .format(end_time - start_time))
        print("")

        # At this point we have the dictionaries that map the oldIDs to the
        # newIDs in addition to the indices that control the sorting of the
        # forests.  We now loop through all the fields within each halo within
        # each snapshot and if the field contains a haloID we update it.
        # While going through each field, we will then write out the data into
        # a new HDF5 file in the order specified by indices.

        print("")
        print("Now writing out the snapshots in the sorted order.")
        start_time = time.time()

        for key in tqdm(f_in.keys()):
            cmn.copy_group(f_in, f_out, key)

            #f_out.create_dataset("oldIDs", 
            #                     list(ID_maps[Snap_Nums[snap_key]].keys()))

            #f_out.create_dataset("newIDs", 
            #                     list(ID_maps[Snap_Nums[snap_key]].values()))


            for field in f_in[key]:

                # Some keys (e.g., 'Header') don't have snapshots so need an
                # except to catch this.
                try:
                    NHalos = len(f_in[key][args["halo_id"]])
                    if (NHalos == 0):
                        continue
                except KeyError:
                    continue

                if field in args["ID_fields"]:  # If this field has an ID...
                    # Need to get the oldIDs, find the snapshot they correspond
                    # to and then get the newIDs using our dictionary.                
                    oldID = f_in[key][field][:]
                    snapnum = cmn.temporalID_to_snapnum(oldID,
                                                        args["index_mult_factor"])
                    newID = [ID_maps[snap][ID] for snap, ID in zip(snapnum,
                                                                   oldID)]
                    to_write = np.array(newID)  # Remember what we need to write.
                else:
                    to_write = f_in[key][field][:]

                # We know what we need to write, so let's write it in the
                # correct order.
                f_out[key][field][:] = to_write[snapshot_indices[key]]

        end_time = time.time()
        print("Writing of snapshots took {0:3f} seconds".
              format(end_time - start_time))
        print("Done!")
        print("")


if __name__ == '__main__':

    args = parse_inputs()
    sort_and_write_file(args)
