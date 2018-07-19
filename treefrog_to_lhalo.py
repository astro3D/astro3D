#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import os.path

import time

from genesis.utils import common as cmn


def get_LHalo_datastruct():
    """
    Generates the LHalo numpy structured array.

    Parameters
    ----------

    None.

    Returns
    ----------

    LHalo_Desc: numpy structured array.  Required.
        Structured array for the LHaloTree data format.
    """

    LHalo_Desc_full = [
        ('Descendant',          np.int32),
        ('FirstProgenitor',     np.int32),
        ('NextProgenitor',      np.int32),
        ('FirstHaloInFOFgroup', np.int32),
        ('NextHaloInFOFgroup',  np.int32),
        ('Len',                 np.int32),
        ('M_Mean200',           np.float32),
        ('Mvir',                np.float32),
        ('M_TopHat',            np.float32),
        ('Posx',                np.float32), 
        ('Posy',                np.float32), 
        ('Posz',                np.float32), 
        ('Velx',                np.float32), 
        ('Vely',                np.float32), 
        ('Velz',                np.float32), 
        ('VelDisp',             np.float32),
        ('Vmax',                np.float32),
        ('Spinx',               np.float32),
        ('Spiny',               np.float32),
        ('Spinz',               np.float32),
        ('MostBoundID',         np.int64),
        ('SnapNum',             np.int32),
        ('Filenr',              np.int32),
        ('SubHaloIndex',        np.int32),
        ('SubHalfMass',         np.float32)
                         ]

    names = [LHalo_Desc_full[i][0] for i in range(len(LHalo_Desc_full))]
    formats = [LHalo_Desc_full[i][1] for i in range(len(LHalo_Desc_full))]
    LHalo_Desc = np.dtype({'names': names, 'formats': formats}, align=True)

    return LHalo_Desc


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
    parser.add_argument("-b", "--fname_out_binary", dest="fname_out_binary",
                        help="Path to the output LHaloTree binary data file. "
                             "Required.") 
    parser.add_argument("-t", "--forestID", dest="forest_id",
                        help="Field name for forest/tree ID. Default: ForestID.",
                        default="ForestID")
    parser.add_argument("-i", "--HaloID", dest="halo_id",
                        help="Field name for halo ID. Default: ID.",
                        default="ID")
    parser.add_argument("-d", "--ID_fields", dest="ID_fields",
                        help="Field names for those that contain IDs.  "
                        "Separate field names with a comma. "
                        "Default: ID,Tail,Head,NextProgenitor,NextSubhalo,"
                        "PreviousProgenitor,PreviousSubhalo,RootHead,RootTail,hostHaloID",
                        default=("ID,Tail,Head,NextProgenitor,NextSubhalo,"
                                 "PreviousProgenitor,PreviousSubhalo,RootHead,"
                                 "RootTail,hostHaloID"))
    parser.add_argument("-x", "--index_mult_factor", dest="index_mult_factor",
                        help="Conversion factor to go from a unique, "
                        "per-snapshot halo index to a temporally unique haloID."
                        " Default: 1e12.", default=1e12)
    parser.add_argument("-p", "--fofID", dest="fofID",
                        help="Field name that contains the "
                        "FirstHaloInFoFgroup ID. Default: hostHaloID.",
                        default="hostHaloID")
    parser.add_argument("-c", "--convert_indices", dest="convert_indices",
                        help="Flag to control if we should convert the input "
                             "file IDs to be snapshot local. Default: 1.",
                        default=1, type=int)
 
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


def fix_flybys(tree):
    """
    Fixes flybys for a single forest. 

    Under the LHalo tree data structure, multiple FoFs at the root redshift are
    allowed IF AND ONLY IF all `FirstHaloInFoFgroup` values point to the same
    FoF. This is not enforced in the Treefrog data structure hence we must fix it
    here.

    We designate the most massive FoF halo at the root redshift to be the
    'True' FoF halo and update the Treefrog-equivalent field of
    `FirstHaloInFoFgroup` to point to this most massive halo. 

    Parameters
    ----------

    f_in, f_out: Open HDF5 files. Required.
        Opened HDF5 files that we are reading from/writing to.

    snap_key: String. Required.
        Field name associated with the root snapshot.

    snap_num: Integer.  Required.
        Snapshot number of the root snapshot.

    IDmap: Nested dictionary keyed by the snapshot number of HaloID.  Required.
        Map of the old, global halo IDs to the new forest-local IDs.

    args: Dictionary.  Required.
        Dictionary containing the argsion parameters specified at runtime.

    NHalos_Forest: Nested Dictionary. Required.
        Nested dictionary that contains the number of halos for each Forest at
        each snapshot.  Outer-key is the ForestID and inner-key is the snapshot
        key.

    NHalos_Forest_Offset: Nested Dictionary. Required.
        Nested dictionary that contains the offset for each Forest at each
        snapshot. Outer-key is the ForestID and inner-key is the snapshot key.
        This is required because whilst the tree is sorted by ForestID, the
        relative position of the tree can change from snapshot to snapshot.

    Returns
    ----------

    None.

    Since `f_out` is an opened HDF5 file in write mode, the fixed values are
    written to disk on-the-fly (pun intended). 
    """

    max_snapnum = max(tree["SnapNum"])
    all_indices = np.arange(len(tree))
    root_fof_halo_inds = (np.where((tree["SnapNum"] == max_snapnum) &
                               (tree["FirstHaloInFOFgroup"] == all_indices)
                              ))[0]
    # if there is only one FOF halo, 
    # no changes need to be made
    if len(root_fof_halo_inds) == 1:
        return tree

    max_fof_mass_idx = np.argmax(tree[mass_field][root_fof_halo_inds])
    if fof_idx == max_fof_mass_idx:
        next_subhalo = tree["NextHaloInFOFgroup"][fof_idx]
        while next_subhalo != next_subhalo:
            next_subhalo = tree["NextHaloInFOFgroup"][next_subhalo]

    #for fof_idx in root_fof_halo_inds:
    

    #tree["FirstHaloInFOFgroup"][rooot_fofs_indices

    true_fof_idx = np.argmax(tree["Mvir"][root_fofs])
    true_fof = tree["FirstHaloInFOFgroup"][true_fof_idx] 

    tree["FirstHaloInFOFgroup"][root_fofs] = true_fof

    return tree


def convert_treefrog(args):
    """
    Converts Treefrog trees into LHalo trees.

    The data-structure of the Treefrog trees is assumed to be HDF5 File ->
    Snapshots -> Halo Properties at each snapshot. The LHalo tree is saved as a
    separate HDF5 file.

    Parameters
    ----------

    args: Dictionary.  Required.
        Dictionary containing the argsion parameters specified at runtime.

    Returns
    ----------

    None.
    """

    with h5py.File(args["fname_in"], "r") as f_in, \
         h5py.File(args["fname_out"], "w") as f_out:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        NHalos_Forest, NHalos_Forest_Offset = cmn.get_halos_per_forest(f_in,
                                                                       Snap_Keys,
                                                                       args)

        print("Copying the old file to a new one.")            
        for key in tqdm(f_in.keys()):
            cmn.copy_group(f_in, f_out, key, args)

        print("Now creating a dictionary that maps the old, global indices to " 
              "ones that are forest-local.")

        NHalos_processed = np.zeros(len(NHalos_Forest.keys()))
        Forests_InSnap = {}
        ID_maps = {}
        for snap_key in tqdm(Snap_Keys[::-1]):
            try:
                NHalos = len(f_in[snap_key][args["halo_id"]])
                if (NHalos == 0):
                    continue
            except KeyError:
                continue

            oldIDs_global = []
            newIDs_global = []

            forests_thissnap = \
            np.unique(f_in[snap_key][args["forest_id"]][:])

            Forests_InSnap[snap_key] = forests_thissnap

            oldIDs = f_in[snap_key][args["halo_id"]][:]

            for forest in forests_thissnap:

                NHalos_snapshot = NHalos_Forest[forest][snap_key]
                offset = NHalos_Forest_Offset[forest][snap_key]

                idx_lower = offset
                idx_upper = NHalos_snapshot + offset

                oldIDs_thisforest = oldIDs[idx_lower:idx_upper] 
                newIDs_thisforest = np.arange(NHalos_processed[forest-1],
                                            NHalos_processed[forest-1] + NHalos_snapshot)

                for val1, val2 in zip(oldIDs_thisforest, newIDs_thisforest):
                    oldIDs_global.append(int(val1))
                    newIDs_global.append(int(val2))

                NHalos_processed[forest-1] += NHalos_snapshot

            oldIDs_to_newIDs = dict(zip(list(oldIDs_global),
                                        list(newIDs_global)))
            ID_maps[Snap_Nums[snap_key]] = oldIDs_to_newIDs

        # For some ID fields (e.g., NextProgenitor), the value is -1.
        # When we convert from the temporalID to a snapshot number, we
        # subtract 1 and divide by the multiplication factor (default 1e12)
        # then cast to an integer.  Hence -2 divided by a huge number will
        # be less than 1 and when it's cast to an integer will result in 0.
        # So the 'Snapshot Number' for values of -1 will be 0.  We want to
        # preserve these -1 flags so we map -1 to -1.
        ID_maps[0] = {-1:-1}
                    
        print("Now going through all the snapshots and updating the IDs.")

        for snap_key in tqdm(Snap_Keys):
            try:
                NHalos = len(f_in[snap_key][args["halo_id"]])
                if (NHalos == 0):
                    continue
            except KeyError:
                continue

            forests_thissnap = \
            np.unique(f_in[snap_key][args["forest_id"]][:])


            for field in args["ID_fields"]:  # If this field has an ID...

                oldID = f_in[snap_key][field][:]
                snapnum = cmn.temporalID_to_snapnum(oldID,
                                                    args["index_mult_factor"])

                # We now want to map the oldIDs to the new, forest-local
                # IDs.  However because you can't hash a dictionary with a
                # numpy array, this needs to be done manually in a `for`
                # loop.

                newID = [ID_maps[snap][ID] for snap, ID in zip(snapnum,  
                                                               oldID)]
                f_out[snap_key][field][:] = newID

            # Field Loop.
        # Snapshot Loop.


def write_out_lhalo_binary(fname_in, fname_out, halo_id_name, forest_id_name):
    
    # Find the max value of the object where the compared
    # values are returned via the "key". In this case, 
    # compares the integer Snapshot number values, and then 
    # returns the "Snapshot_group" key in the Snap_Keys dictionary.
    # Taken from
    # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

    LHalo_Desc = get_LHalo_datastruct()
   
    with h5py.File(fname_in, "r") as f_in:
        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        NHalos_Forest, NHalos_Forest_Offset = cmn.get_halos_per_forest(f_in,
                                                                       Snap_Keys,
                                                                       halo_id_name,
                                                                       forest_id_name)
        last_snap_key = max(Snap_Nums, key=Snap_Nums.get)

        forest_to_process = np.unique(f_in[last_snap_key][forest_id_name][:])
        for ForestID in forest_to_process:
            NHalos = sum(NHalos_Forest[ForestID].values())
            print("For Forest {0} there are {1} halos.".format(ForestID,
                  NHalos))

            tree = np.zeros(NHalos, dtype=LHalo_Desc)
            offset = 0

            for snap_key in Snap_Keys[::-1]: 
                halos_forest_snap = list(np.where(f_in[snap_key][forest_id_name][:] == ForestID)[0])
                print("Filling {1} halos for Snapshot {0}".format(snap_key, len(halos_forest_snap)))
                if len(halos_forest_snap) > 0:
                    tree, offset = fill_LHalo_properties(tree, f_in[snap_key], 
                                                         halos_forest_snap, 
                                                         offset,
                                                         Snap_Nums[snap_key], 0)
                    print(tree)
                    print(tree["FirstHaloInFOFgroup"])
                    exit()

            all_descendants = tree["Descendant"][:]
            for ii, d in enumerate(all_descendants):
                curr = tree["FirstProgenitor"][d]
                while tree["NextProgenitor"][curr] != -1:
                    curr = tree["NextProgenitor"][curr]

                tree["NextProgenitor"][curr] = ii

            tree = fix_flybys(tree)

            exit()

def fill_LHalo_properties(tree, f_in, halo_indices, current_offset, snapnum,
                          filenr):

    NHalos_thissnap = len(halo_indices)

    tree["Descendant"][current_offset:current_offset+NHalos_thissnap] = f_in["Head"][halo_indices]
    tree["FirstProgenitor"][current_offset:current_offset+NHalos_thissnap]  = f_in["Tail"][halo_indices]
    tree["NextProgenitor"][current_offset:current_offset+NHalos_thissnap]  = -1

    tree["FirstHaloInFOFgroup"][current_offset:current_offset+NHalos_thissnap] = f_in["hostHaloID"][halo_indices]
    tree["NextHaloInFOFgroup"][current_offset:current_offset+NHalos_thissnap] = -1

    all_hosthalo_inds = f_in["hostHaloID"][halo_indices]
    _, sub_and_host_inds = np.unique(all_hosthalo_inds, return_inverse=True)
    for ii in sub_and_host_inds:
        fof_ind = tree["FirstHaloInFOFgroup"][current_offset + ii]
        if fof_ind == current_offset + ii:
            continue

        curr = fof_ind
        while tree["NextHaloInFOFgroup"][current_offset + curr] != -1:
            curr = tree["NextHaloInFOFgroup"][current_offset + curr]

        tree["NextHaloInFOFgroup"][current_offset + curr] = current_offset + ii
            

    tree["Len"][current_offset:current_offset+NHalos_thissnap] = f_in["npart"][halo_indices]
    tree["M_Mean200"][current_offset:current_offset+NHalos_thissnap] = f_in["Mass_200mean"][halo_indices]
    tree["Mvir"][current_offset:current_offset+NHalos_thissnap] = f_in["Mass_200crit"][halo_indices] 
    tree["M_TopHat"][current_offset:current_offset+NHalos_thissnap] = f_in["Mass_200crit"][halo_indices] 

    tree["Posx"][current_offset:current_offset+NHalos_thissnap] = f_in["Xc"][halo_indices]
    tree["Posy"][current_offset:current_offset+NHalos_thissnap] = f_in["Yc"][halo_indices]
    tree["Posz"][current_offset:current_offset+NHalos_thissnap] = f_in["Zc"][halo_indices]

    tree["Velx"][current_offset:current_offset+NHalos_thissnap] = f_in["VXc"][halo_indices]
    tree["Vely"][current_offset:current_offset+NHalos_thissnap] = f_in["VYc"][halo_indices]
    tree["Velz"][current_offset:current_offset+NHalos_thissnap] = f_in["VZc"][halo_indices]

    tree["VelDisp"][current_offset:current_offset+NHalos_thissnap] = f_in["sigV"][halo_indices]
    tree["Vmax"][current_offset:current_offset+NHalos_thissnap]  = f_in["Vmax"][halo_indices]

    tree["Spinx"][current_offset:current_offset+NHalos_thissnap] = f_in["Lx"][halo_indices]
    tree["Spiny"][current_offset:current_offset+NHalos_thissnap] = f_in["Ly"][halo_indices]
    tree["Spinz"][current_offset:current_offset+NHalos_thissnap] = f_in["Lz"][halo_indices]
    
    tree["MostBoundID"] [current_offset:current_offset+NHalos_thissnap]= f_in["ID"][halo_indices] ##

    tree["SnapNum"][current_offset:current_offset+NHalos_thissnap]= snapnum 
    tree["Filenr"][current_offset:current_offset+NHalos_thissnap] = filenr 
       
    tree["SubHaloIndex"][current_offset:current_offset+NHalos_thissnap] = -1 ## 
    tree["SubHalfMass"][current_offset:current_offset+NHalos_thissnap] = -1 ## 

    current_offset += NHalos_thissnap

    return tree, current_offset

 
if __name__ == '__main__':

    args = parse_inputs()

    if args["convert_indices"]:
        convert_treefrog(args)
    else:
        write_out_lhalo_binary(args["fname_out"], args["fname_out_binary"], 
                               args["halo_id"], args["forest_id"]) 

