"""
Authors: Jacob Seiler, Manodeep Sinha
"""

#!/usr/bin:env python
from __future__ import print_function
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import time

from genesis.utils import common as cmn


def get_sort_indices(file_in, snap_key, sort_fields, sort_direction):
    """
    Gets the indices that will sort the HDF5 file.

    This sorting uses the fields provided by the user. The sort fields
    (or sort keys) we ordered such that the first key will peform the
    outer-most sort and the last key will perform the inner-most sort.

    Example:
        sort_fields = ["ForestID", "Mass_200mean"]
        ForestID = [1, 4, 39, 1, 1, 4]
        Mass_200mean = [4e9, 10e10, 8e8, 7e9, 3e11, 5e6]

        Then the indices would be [0, 3, 4, 5, 1, 2]

    Parameters
    ----------

    file_in: HDF5 file.
        Open HDF5 file that we are sorting for. The data structure is assumed
        to be HDF5_File -> Snapshot_Keys -> Halo properties.

    snap_key: String.
        The snapshot field name for the snapshot we are accessing.

    sort_fields: List of strings. 
        List containing the field names we are sorting on.

        ..note::
            The order of this sorting is such that the first key will perform
            the outer-most sort and the last key will perform the inner-most
            sort.

    Returns
    ----------

    indices: `~numpy.ndarray` of integers. 
        Array containing the indices that sorts the data using the specified
        sort keys.
    """

    sort_keys = []

    # We need to reverse `sort_fields` due to the behaviour of `~np.lexsort`. 
    for key, direction in zip(reversed(sort_fields), reversed(sort_direction)):
        if key is None or "NONE" in key.upper():
            continue
        if direction == -1: 
            sort_keys.append(-np.array(file_in[snap_key][key]))
        else:
            sort_keys.append(np.array(file_in[snap_key][key]))

    indices = np.lexsort((sort_keys))

    return indices


def forest_sorter(fname_in, fname_out, haloID_field="ID",
                  sort_fields=["ForestID", "hostHaloID", "Mass_200mean"],
                  sort_direction=np.array([1,1,-1]),
                  ID_fields=["Head", "Tail", "RootHead", "RootTail",
                             "ID", "hostHaloID"], index_mult_factor=1e12):
    """
    Sorts and saves a HDF5 tree file on the specified sort fields.  The IDs of 
    the halos are assume to use the index within the data file and hence will 
    be updated to reflect the sorted order. 

    ..note::
        The default parameters are chosen to match the ASTRO3D Genesis trees as
        produced by VELOCIraptor + Treefrog.    

    Parameters
    ----------

    fname_in, fname_out: String.
        Path to the input HDF5 trees and path to where the sorted trees will be
        saved. 

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
        `common.index_to_temporalID()`.

    Returns
    ----------

    None.
    """

    with h5py.File(fname_in, "r") as f_in, \
         h5py.File(fname_out, "w") as f_out:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        ID_maps = dict()
        snapshot_indices = dict()

        print("")
        print("Generating the dictionary to map the oldIDs to the newIDs.")

        start_time = time.time()
        for snap_key in tqdm(Snap_Keys):
            # We only want to go through snapshots that contain halos.
            if len(f_in[snap_key][haloID_field]) == 0:
                continue

            # Need to get the indices that sort the data according to the
            # specified keys.

            indices = get_sort_indices(f_in, snap_key, sort_fields,
                                       sort_direction)

            old_haloIDs = f_in[snap_key][haloID_field][:]
            old_haloIDs_sorted = old_haloIDs[indices]

            # The ID of a halo depends on its snapshot-local index.
            # As the new haloIDs will be sorted correctly, their index will
            # simply be np.arange(len(Number of Halos)).
            new_haloIDs = cmn.index_to_temporalID(np.arange(len(indices)),
                                                  Snap_Nums[snap_key],
                                                  index_mult_factor)

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
                    NHalos = len(f_in[key][haloID_field])
                    if (NHalos == 0):
                        continue
                except KeyError:
                    continue

                if field in ID_fields:  # If this field has an ID...
                    # Need to get the oldIDs, find the snapshot they correspond
                    # to and then get the newIDs using our dictionary.                
                    oldID = f_in[key][field][:]
                    snapnum = cmn.temporalID_to_snapnum(oldID,
                                                        index_mult_factor)
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
