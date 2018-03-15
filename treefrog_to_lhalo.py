#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py
from tqdm import tqdm
from optparse import OptionParser

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

    opt: optparse.Values.  Required.
        Values from the OptionParser package.  Values are accessed through
        ``opt.Value`` and cast into a dictionary using ``vars(opt)``.
    """

    parser = OptionParser()

    parser.add_option("-f", "--fname_in", dest="fname_in",
                      help="Path to the input HDF5 data file. Required.")
    parser.add_option("-o", "--fname_out", dest="fname_out",
                      help="Path to the output HDF5 data file. Required.")
    parser.add_option("-i", "--HaloID", dest="halo_id",
                      help="Field name for halo ID. Default: ID.",
                      default="ID")
    parser.add_option("-t", "--forestID", dest="forest_id",
                      help="Field name for forest/tree ID. Default: ForestID.",
                      default="ForestID")
    parser.add_option("-p", "--ID_fields", dest="ID_fields",
                      help="Field names for those that contain IDs.  Default: "
                      "('ID', 'Tail', 'Head', 'NextSubHalo', 'Dummy1', "
                      "'Dumm2').",
                      default=('ID', 'Tail', 'Head', 'NextSubHalo', 'Dummy',
                               'Dummy'))
    parser.add_option("-x", "--index_mult_factor", dest="index_mult_factor",
                      help="Conversion factor to go from a unique, "
                      "per-snapshot halo index to a temporally unique haloID. "
                      "Default: 1e12.", default=1e12)

    (opt, args) = parser.parse_args()

    # We require an input file and an output one.
    if (opt.fname_in is None or opt.fname_out is None):
        parser.print_help()
        raise RuntimeError

    # Print some useful startup info. #
    print("")
    print("The HaloID field for each halo is '{0}'.".format(opt.halo_id))
    print("")

    return opt


def get_halos_per_forest(f_in, Snap_Keys, opt):
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

    opt: Dictionary.  Required.
        Dictionary containing the option parameters specified at runtime.
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

    for count, snap_key in enumerate(tqdm(Snap_Keys)):
        if len(f_in[snap_key][opt["halo_id"]]) == 0:  # Skip empty snapshots.
            continue

        forestIDs = np.unique(f_in[snap_key][opt["forest_id"]])    

        halo_forestids = f_in[snap_key][opt["forest_id"]][:]
        for forest_id in forestIDs:
            ThisSnap_NHalos = len(np.where(halo_forestids
                                                  == forest_id)[0])

            ThisSnap_NHalos_Forest = {snap_key : ThisSnap_NHalos} 
          
            try:           
                NHalos_Forest[forest_id][snap_key] = ThisSnap_NHalos
            except KeyError:
                NHalos_Forest[forest_id] = {snap_key : ThisSnap_NHalos} 
            if (forest_id == 524288):
                print(sorted(NHalos_Forest[524288]))
                print(snap_key)
        if count > 80:
            break

    print("Done")
    return NHalos_Forest


def convert_treefrog(opt):
    """

    """

    with h5py.File(opt["fname_in"], "r") as f_in, \
         h5py.File(opt["fname_out"], "w") as f_out:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        start_time = time.time()

        NHalos_Forest = get_halos_per_forest(f_in, Snap_Keys, opt)

        print("Now building the dictionary that maps the oldIDs to a "
              "forest-local index.")

        # Loop through each forest.
        # Then for each forest loop through each snapshot.
        # Keep a record of how many halos have been processed for each forest.
        # Maybe then keep a record of how many halos have been processed for
        # each snapshot?

        oldIDs_to_newIDs = {}
        NHalos_processed_snapshot = dict(zip(Snap_Keys,
                                         np.zeros(len(Snap_Keys))))
        

        for forest in tqdm(NHalos_Forest.keys()):
            NHalos_processed = 0
            print(forest) 
            print(sorted(NHalos_Forest[forest].keys()))

            for snap_key in sorted(NHalos_Forest[forest].keys()):
                NHalos_snapshot = NHalos_Forest[forest][snap_key]
                idx_lower = NHalos_processed
                idx_upper = NHalos_processed + NHalos_snapshot

                oldIDs = f_in[snap_key][opt["halo_id"]][idx_lower:idx_upper]
                newIDs = np.arange(NHalos_processed, 
                                   NHalos_processed + NHalos_snapshot)

                print("For forest {0} at snapshot {1} the oldIDs are "
                        "{2}".format(forest, snap_key, oldIDs))

                ID_map = dict(zip(oldIDs, newIDs))
                NHalos_processed_snapshot[forest][snap_key] += 1

                oldIDs_to_newIDs[snap_key] = ID_map
                print(oldIDs_to_newIDs)
                print(ID_map)
            exit()


if __name__ == '__main__':

    opt = parse_inputs()
    convert_treefrog(vars(opt))
