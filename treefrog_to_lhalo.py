#!/usr/bin/env python
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
    parser.add_argument("-t", "--forestID", dest="forest_id",
                        help="Field name for forest/tree ID. Default: ForestID.",
                        default="ForestID")
    parser.add_argument("-i", "--HaloID", dest="halo_id",
                        help="Field name for halo ID. Default: ID.",
                        default="ID")
    parser.add_argument("-p", "--ID_fields", dest="ID_fields",
                        help="Field names for those that contain IDs.  "
                        "Separate field names with a comma. "
                        "Default: ID,Tail,Head,NextProgenitor,NextSubhalo,"
                        "PreviousProgenitor,PreviousSubhalo,RootHead,RootTail",
                        default=("ID,Tail,Head,NextProgenitor,NextSubhalo,"
                        "PreviousProgenitor,PreviousSubhalo,RootHead,RootTail"))
    parser.add_argument("-x", "--index_mult_factor", dest="index_mult_factor",
                        help="Conversion factor to go from a unique, "
                        "per-snapshot halo index to a temporally unique haloID. "
                        "Default: 1e12.", default=1e12)

    args = parser.parse_args()

    # We allow the user to enter an arbitrary number of sort fields and fields
    # that contain IDs.  They are etnered as a single string separated by
    # commas so need to split them up into a list.
    args.ID_fields = (args.ID_fields).split(',')

    # We require an input file and an output one.
    if (args.fname_in is None or args.fname_out is None):
        parser.print_help()
        raise RuntimeError

    # Print some useful startup info. #
    print("")
    print("The HaloID field for each halo is {0}.".format(args.halo_id))
    print("The fields that contain IDs are {0}".format(args.ID_fields))
    print("")

    return vars(args)


def get_halos_per_forest(f_in, Snap_Keys, args):
    """
    Determines the number of halos in each forest.

    The resulting Dictionary is nested with the outer-key given by the ForestID
    and the innter-key given by the snapshot field name.

    E.g., If Forest 5 has 10 halos at Snapshot 20 and 100 at Snapshot 21 then,
        NHalos_Forest[5]['Snap_020'] = 10
        NHalos_Forest[5]['Snap_021'] = 100

    Parameters
    ----------

    f_in: Open HDF5 file. Required.
        HDF5 file that contains the sorted data.

    Snap_Keys: List. Required.
        List of keys that correspond to the fields containing the snapshot
        data.

    args: Dictionary.  Required.
        Dictionary containing the argsion parameters specified at runtime.
        Used to specify field name for the ForestID.

    Returns
    ----------

    NHalos_Forest: Nested Dictionary. Required.
        Nested dictionary that contains the number of halos for each Forest at
        each snapshot.  Outer-key is the ForestID and inner-key is the snapshot
        key.
    """

    print("")
    print("Generating the dictionary for the number of halos in each tree "
          "at each snapshot.")

    NHalos_Forest = {}
    NHalos_Forest_Offset = {}

    for count, snap_key in enumerate(tqdm(Snap_Keys)):
        if len(f_in[snap_key][args["halo_id"]]) == 0:  # Skip empty snapshots.
            continue
        
        halos_counted = 0
        halo_forestids = f_in[snap_key][args["forest_id"]][:]

        # First get the number of halos in each forest then grab the indices
        # (i.e., the forestID as we start from 0) of the forests that have
        # halos.
        forests_binned = np.bincount(halo_forestids)
        forestIDs = np.nonzero(forests_binned)[0]
 
        for forest_id in forestIDs:
            this_snap_NHalos = forests_binned[forest_id] 
            this_snap_NHalos_forest = {snap_key : this_snap_NHalos} 
         
            try:           
                NHalos_Forest[forest_id][snap_key] = this_snap_NHalos
                NHalos_Forest_Offset[forest_id][snap_key] = halos_counted
            except KeyError:
                NHalos_Forest[forest_id] = {snap_key : this_snap_NHalos}
                NHalos_Forest_Offset[forest_id] = {snap_key : halos_counted}
   
            halos_counted += this_snap_NHalos

    print("Done")
    return NHalos_Forest, NHalos_Forest_Offset


def plot_forests(NHalos_Forest):
    """
    """
    import matplotlib
    import pylab as plt

    ax1 = plt.subplot(111)

    NHalos = []
    print("Plotting some forest statistics.")

    for forest in tqdm(NHalos_Forest.keys()):
        NHalos_this_forest = 0
        for snap_key in NHalos_Forest[forest].keys(): 
           NHalos_this_forest += NHalos_Forest[forest][snap_key]

        NHalos.append(NHalos_this_forest)

    ax1.scatter(np.arange(len(NHalos_Forest)), NHalos)

    ax1.set_xlabel("Forest index")
    ax1.set_ylabel("Number of Halos")
    ax1.set_yscale("log")

    output_file = "./forest.png"
    plt.savefig(output_file)

    plt.close()

def convert_treefrog(args):
    """

    """

    with h5py.File(args["fname_in"], "r") as f_in, \
         h5py.File(args["fname_out"], "w") as f_out:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        start_time = time.time()

        NHalos_Forest, NHalos_Forest_Offset = get_halos_per_forest(f_in, 
                                                                   Snap_Keys,
                                                                   args)

        plot_forests(NHalos_Forest)
        exit()
        print("Now building the dictionary that maps the oldIDs to a "
              "forest-local index.")

        # Loop through each forest.
        # Then for each forest loop through each snapshot.
        # Keep a record of how many halos have been processed for each forest.
        # Maybe then keep a record of how many halos have been processed for
        # each snapshot?

        NHalos_processed_snapshot = dict(zip(Snap_Keys,
                                         np.zeros((len(Snap_Keys)))))
        

        # Let's copy everything into a new file so we don't make any mistakes.
        print("Copying the old file to a new one.")
        for key in tqdm(f_in.keys()):
            cmn.copy_group(f_in, f_out, key, args)


        for count, forest in enumerate(tqdm(NHalos_Forest.keys())):
            NHalos_processed = 0  # Number of halos in this forest that have
                                  # been processed.
            ID_maps = {}
            for snap_key in sorted(NHalos_Forest[forest].keys()):
                NHalos_snapshot = NHalos_Forest[forest][snap_key]
                offset = NHalos_Forest_Offset[forest][snap_key]

                idx_lower = offset
                idx_upper = NHalos_snapshot + offset

                oldIDs = f_in[snap_key][args["halo_id"]][idx_lower:idx_upper]
                newIDs = np.arange(NHalos_processed, 
                                   NHalos_processed + NHalos_snapshot)

                #print("For forest {0} at snapshot {1} there are {2} oldIDs" 
                #      .format(forest, snap_key, len(oldIDs)))

                oldIDs_to_newIDs = dict(zip(oldIDs, newIDs))
                ID_maps[Snap_Nums[snap_key]] = oldIDs_to_newIDs

                NHalos_processed += NHalos_snapshot 

            # For some ID fields (e.g., NextProgenitor), the value is -1.  When we
            # convert the temporalID to a Snapshot number, the operation we
            # do is
            # +1 * mult_factor.
            # If the ID is -1 the 'Snapshot number' will be
            # 0.  We want to preserve
            # these -1 so we map to itself.
            ID_maps[0] = {-1:-1}

            # We now have the ID mapping for this forest.  Let's loop back
            # through all the snapshots and update the ID to be forest-local.
            for snap_key in sorted(NHalos_Forest[forest].keys()):    
                for field in args["ID_fields"]:  # If this field has an ID...
                    NHalos_snapshot = NHalos_Forest[forest][snap_key]
                    offset = NHalos_Forest_Offset[forest][snap_key]

                    idx_lower = offset 
                    idx_upper = NHalos_snapshot + offset

                    oldID = f_in[snap_key][field][idx_lower:idx_upper]
                    snapnum = cmn.temporalID_to_snapnum(oldID,
                                                        args["index_mult_factor"])
                    #print(oldID)
                    #print(ID_maps[18])
                    newID = [ID_maps[snap][ID] for snap, ID in zip(snapnum,
                                                                   oldID)]

                    f_out[snap_key][field][idx_lower:idx_upper] = newID 
            if count > 200:
                exit()

if __name__ == '__main__':

    args = parse_inputs()
    convert_treefrog(args)
