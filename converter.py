"""
Authors: Jacob Seiler, Manodeep Sinha
"""

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


def fix_nextprog(tree):

    all_descendants = tree["Descendant"][:]
    for ii, d in enumerate(all_descendants):
        curr = tree["FirstProgenitor"][d]
        while tree["NextProgenitor"][curr] != -1:
            curr = tree["NextProgenitor"][curr]

        tree["NextProgenitor"][curr] = ii

    return tree

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

    NHalos_forest: Nested Dictionary. Required.
        Nested dictionary that contains the number of halos for each Forest at
        each snapshot.  Outer-key is the ForestID and inner-key is the snapshot
        key.

    NHalos_forest_offset: Nested Dictionary. Required.
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


def convert_indices(fname_in, fname_out, 
                    haloID_field="ID", forestID_field="ForestID", 
                    ID_fields=["Head", "Tail", "RootHead", "RootTail", "ID", 
                               "hostHaloID"], index_mult_factor=1e12):
    """
    Converts temporally unique tree IDs to ones that are forest-local as 
    required by the LHalo Trees format. 

    The data-structure of the Treefrog trees is assumed to be HDF5 File ->
    Snapshots -> Halo Properties at each snapshot.

    A new HDF5 file is saved out with the updated IDs.

    ..note::
        We require the input trees to be sorted via the forest ID 
        (`forestID_field`) and suggest to also sub-sort on hostHaloID and mass.
        Sorting can be done using the `forest_sorter()` function.

    ..note::
        The default parameters are chosen to match the ASTRO3D Genesis trees as
        produced by VELOCIraptor + Treefrog.
 
    Parameters
    ----------

    fname_in, fname_out: String.
        Path to the input HDF5 VELOCIraptor + treefrog trees and the path 
        where the LHalo correct trees will be saved.

    haloID_field: String. Default: 'ID'.
        Field name within the HDF5 file that corresponds to the unique halo ID.

    forestID_field: String. Default: 'ForestID'.
        Field name within the HDF5 file that corresponds to forest ID. 

    ID_fields: List of string. Default: ['Head', 'Tail', 'RootHead', 'RootTail',
                                        'ID', 'hostHaloID'].
        The HDF5 field names that correspond to properties that use halo IDs.
        As the halo IDs are updated to match the required LHalo Tree format,
        these must also be updated. 

    index_mult_factor: Integer. Default: 1e12.
        Multiplication factor to generate a temporally unique halo ID. See
        `common.index_to_temporalID()`.

    Returns
    ----------

    None.
    """

    print("Converting the temporally unique IDs to ones that are forest "
          "local.")

    with h5py.File(fname_in, "r") as f_in, \
         h5py.File(fname_out, "w") as f_out:

        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        NHalos_forest, NHalos_forest_offset = cmn.get_halos_per_forest(f_in,
                                                                       Snap_Keys,
                                                                       haloID_field,
                                                                       forestID_field)

        print("Copying the old tree file to a new one.")
        for key in tqdm(f_in.keys()):
            cmn.copy_group(f_in, f_out, key)

        print("Now creating a dictionary that maps the old, global indices to " 
              "ones that are forest-local.")

        start_time = time.time()

        NHalos_processed = np.zeros(len(NHalos_forest.keys()))
        Forests_InSnap = {}
        ID_maps = {}
        for snap_key in tqdm(Snap_Keys[::-1]):
            try:
                NHalos = len(f_in[snap_key][haloID_field])
                if (NHalos == 0):
                    continue
            except KeyError:
                continue

            oldIDs_global = []
            newIDs_global = []

            forests_thissnap = \
            np.unique(f_in[snap_key][forestID_field][:])

            Forests_InSnap[snap_key] = forests_thissnap

            oldIDs = f_in[snap_key][haloID_field][:]

            for forest in forests_thissnap:

                NHalos_snapshot = NHalos_forest[forest][snap_key]
                offset = NHalos_forest_offset[forest][snap_key]

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
                    
        end_time = time.time()
        print("Creation of dictionary took {0:3f} seconds." \
              .format(end_time - start_time))

        print("Now going through all the snapshots and updating the IDs.")
        start_time = time.time()

        for snap_key in tqdm(Snap_Keys):
            try:
                NHalos = len(f_in[snap_key][haloID_field])
                if (NHalos == 0):
                    continue
            except KeyError:
                continue

            forests_thissnap = \
            np.unique(f_in[snap_key][forestID_field][:])


            for field in ID_fields:  # If this field has an ID...

                oldID = f_in[snap_key][field][:]
                snapnum = cmn.temporalID_to_snapnum(oldID,
                                                    index_mult_factor)

                # We now want to map the oldIDs to the new, forest-local
                # IDs.  However because you can't hash a dictionary with a
                # numpy array, this needs to be done manually in a `for`
                # loop.

                newID = [ID_maps[snap][ID] for snap, ID in zip(snapnum,  
                                                               oldID)]
                f_out[snap_key][field][:] = newID

            # Field Loop.
        # Snapshot Loop.

        end_time = time.time()
        print("Updating took {0:.3f} seconds.".format(end_time - start_time))


def create_lhalo_binary(fname_in, fname_out, haloID_field="ID", 
                        forestID_field="ForestID"): 
    """
    Converts temporally unique tree IDs to ones that are forest-local as 
    required by the LHalo Trees format. 

    The data-structure of the Treefrog trees is assumed to be HDF5 File ->
    Snapshots -> Halo Properties at each snapshot.

    A new HDF5 file is saved out with the updated IDs.

    ..note::
        We require the input trees to be sorted via the forest ID 
        (`forestID_field`) and suggest to also sub-sort on hostHaloID and mass.
        Sorting can be done using the `forest_sorter()` function.

    ..note::
        The default parameters are chosen to match the ASTRO3D Genesis trees as
        produced by VELOCIraptor + Treefrog.
 
    Parameters
    ----------

    fname_in, fname_out: String.
        Path to the input HDF5 VELOCIraptor + treefrog trees and the path 
        where the LHalo correct trees will be saved.

    haloID_field: String. Default: 'ID'.
        Field name within the HDF5 file that corresponds to the unique halo ID.

    forestID_field: String. Default: 'ForestID'.
        Field name within the HDF5 file that corresponds to forest ID. 

    ID_fields: List of string. Default: ['Head', 'Tail', 'RootHead', 'RootTail',
                                        'ID', 'hostHaloID'].
        The HDF5 field names that correspond to properties that use halo IDs.
        As the halo IDs are updated to match the required LHalo Tree format,
        these must also be updated. 

    index_mult_factor: Integer. Default: 1e12.
        Multiplication factor to generate a temporally unique halo ID. See
        `common.index_to_temporalID()`.

    Returns
    ----------

    None.
    """
 
    print("Writing out the trees that are already sorted and have tree-local "
          "indices in the LHalo binary format.")
  
    LHalo_Desc = get_LHalo_datastruct()
   
    with h5py.File(fname_in, "r") as f_in:
        Snap_Keys, Snap_Nums = cmn.get_snapkeys_and_nums(f_in.keys())

        NHalos_forest, NHalos_forest_offset = cmn.get_halos_per_forest(f_in,
                                                                       Snap_Keys,
                                                                       haloID_field, 
                                                                       forestID_field)

        # Find the max value of the object where the compared
        # values are returned via the "key". In this case, 
        # compares the integer Snapshot number values, and then 
        # returns the "Snapshot_group" key in the Snap_Keys dictionary.
        # Taken from
        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
  
        last_snap_key = max(Snap_Nums, key=Snap_Nums.get)

        forests_to_process = np.unique(f_in[last_snap_key][forestID_field][:])
        NForests = len(NHalos_forest.keys())
        print("Forests from unique {0}. Forest from keys "
              "{1}".format(len(forests_to_process), NForests))
        
        filenr = 0

        # We first want to determine the number of forests, total number of
        # halos in these forests, and number of halos per forest for each
        # forest we are processing.
        totNHalos = 0
        global_halos_per_forest = []

        for forestID in forests_to_process:
            # NHalos_forest is a nested dictionary accessed by each forestID.            
            halos_per_forest = sum(NHalos_forest[forestID].values())
            global_halos_per_forest.append(halos_per_forest)
            totNHalos += halos_per_forest 

        # Write out the header with all this info.
        print("Writing {0} forests containing a total of {1} halos."\
              .format(len(forests_to_process), totNHalos))
        write_header(fname_out, len(forests_to_process), totNHalos,
                     global_halos_per_forest)

        # Now for each forest we want to populate the LHalos forest struct, fix
        # any IDs (e.g., flybys) and then write them out.
        for forestID in forests_to_process:

            NHalos = sum(NHalos_forest[forestID].values())
            print("For Forest {0} there are {1} halos.".format(forestID,
                  NHalos))

            forest_halos = np.zeros(NHalos, dtype=LHalo_Desc)

            forest_halos = populate_forest(f_in, forest_halos, Snap_Keys,
                                           Snap_Nums, forestID, NHalos_forest,
                                           NHalos_forest_offset, filenr) 
   
            print("Forest fully populated, now fixing up indices.") 
            #forest_halos = fix_nextprog(forest_halos)
            #forest_halos = fix_flybys(forest_halos)

            print("Writing out forest.")
            write_forest(fname_out, forest_halos)
            

def write_header(fname_out, Nforests, totNHalos, halos_per_forest):
    """
    Creates the LHalo Tree binary file and writes the header information. This
    is of the form:

    Number of Forests within this file (Nforests): 4-byte integer.
    Total number of halos within this file: 4-byte integer.
    Number of halos within each forest for this file: Nforests*4-bytes integers
    
    Parameters
    ----------

    fname_out: String.
        Path to where the LHalo tree binary will be saved. 

    Nforest: Integer.
        Number of forests that will be saved in this file.

    totNHalos: Integer.
        Total number of halos that will be saved in this file.
        
    halos_per_forest: List of integers.
        The number of halos within each forest that will be saved in this file.

    Returns
    ----------

    None.
    """

    print("Writing the LHalo binary header.")

    with open(fname_out, "wb") as f_out:
        f_out.write(np.array(Nforests, dtype=np.int32).tobytes())
        f_out.write(np.array(totNHalos, dtype=np.int32).tobytes())
        f_out.write(np.array(halos_per_forest, dtype=np.int32).tobytes())


    return


def populate_forest(f_in, forest_halos, Snap_Keys, Snap_Nums, forestID, 
                    NHalos_forest, NHalos_forest_offset, filenr):
 
    halos_offset = 0  # We need to slice the halos into the forest array in the
                      # proper place.

    # Start at the root redshift and work our way up the tree.
    for snap_key in Snap_Keys[::-1]:

        # Get the number, index offset and the corresponding indices for halos
        # at this snapshot.
        try:
            NHalos_forest_snap = NHalos_forest[forestID][snap_key]
        except KeyError:
            continue    

        halos_forest_offset = NHalos_forest_offset[forestID][snap_key]
        halos_forest_inds = list(np.arange(halos_forest_offset,
                                           halos_forest_offset + NHalos_forest_snap))
        print("Populating {1} halos for Snapshot {0}".format(snap_key, 
                                                             len(halos_forest_inds)))

        forest_halos, halos_offset = fill_LHalo_properties(forest_halos, 
                                                           f_in[snap_key], 
                                                           halos_forest_inds, 
                                                           halos_offset,
                                                           Snap_Nums[snap_key],
                                                           filenr)

    return forest_halos


def fill_LHalo_properties(forest_halos, f_in, halo_indices, current_offset, snapnum,
                          filenr):

    NHalos_thissnap = len(halo_indices)

    '''
    print(halo_indices)
    print(f_in["Head"][halo_indices])
    print(current_offset)
    print(current_offset+NHalos_thissnap)
    print(forest_halos["Descendant"][current_offset:current_offset+NHalos_thissnap])
    '''

    forest_halos["Descendant"][current_offset:current_offset+NHalos_thissnap] = f_in["Head"][halo_indices]
    forest_halos["FirstProgenitor"][current_offset:current_offset+NHalos_thissnap]  = f_in["Tail"][halo_indices]
    forest_halos["NextProgenitor"][current_offset:current_offset+NHalos_thissnap]  = -1

    forest_halos["FirstHaloInFOFgroup"][current_offset:current_offset+NHalos_thissnap] = f_in["hostHaloID"][halo_indices]
    forest_halos["NextHaloInFOFgroup"][current_offset:current_offset+NHalos_thissnap] = -1

    '''
    all_hosthalo_inds = f_in["hostHaloID"][halo_indices]
    _, sub_and_host_inds = np.unique(all_hosthalo_inds, return_inverse=True)
    for ii in sub_and_host_inds:
        fof_ind = forest_halos["FirstHaloInFOFgroup"][current_offset + ii]
        if fof_ind == current_offset + ii:
            assert(forest_halos["FirstHaloInFOFgroup"][fof_ind] == current_offset + ii)
            continue

        if forest_halos["NextHaloInFOFgroup"][fof_ind] == -1:
            forest_halos["NextHaloInFOFgroup"][fof_ind] = current_offset + ii
            continue
        curr = fof_ind - current_offset
        while forest_halos["NextHaloInFOFgroup"][current_offset + curr] != -1:
            print("Curr {0}\tii {1}\tcurrent_offset "
                  "{2}\tforest_halos['NextHaloInFOFgroup'][current_offset+curr] {3}"\
                  .format(curr, ii, current_offset,
                          forest_halos["NextHaloInFOFgroup"][current_offset+curr]))
            curr = forest_halos["NextHaloInFOFgroup"][current_offset + curr]

        forest_halos["NextHaloInFOFgroup"][current_offset + curr] = current_offset + ii

    '''

    forest_halos["Len"][current_offset:current_offset+NHalos_thissnap] = f_in["npart"][halo_indices]
    forest_halos["M_Mean200"][current_offset:current_offset+NHalos_thissnap] = f_in["Mass_200mean"][halo_indices]
    forest_halos["Mvir"][current_offset:current_offset+NHalos_thissnap] = f_in["Mass_200crit"][halo_indices] 
    forest_halos["M_TopHat"][current_offset:current_offset+NHalos_thissnap] = f_in["Mass_200crit"][halo_indices] 

    forest_halos["Posx"][current_offset:current_offset+NHalos_thissnap] = f_in["Xc"][halo_indices]
    forest_halos["Posy"][current_offset:current_offset+NHalos_thissnap] = f_in["Yc"][halo_indices]
    forest_halos["Posz"][current_offset:current_offset+NHalos_thissnap] = f_in["Zc"][halo_indices]

    forest_halos["Velx"][current_offset:current_offset+NHalos_thissnap] = f_in["VXc"][halo_indices]
    forest_halos["Vely"][current_offset:current_offset+NHalos_thissnap] = f_in["VYc"][halo_indices]
    forest_halos["Velz"][current_offset:current_offset+NHalos_thissnap] = f_in["VZc"][halo_indices]

    forest_halos["VelDisp"][current_offset:current_offset+NHalos_thissnap] = f_in["sigV"][halo_indices]
    forest_halos["Vmax"][current_offset:current_offset+NHalos_thissnap]  = f_in["Vmax"][halo_indices]

    forest_halos["Spinx"][current_offset:current_offset+NHalos_thissnap] = f_in["Lx"][halo_indices]
    forest_halos["Spiny"][current_offset:current_offset+NHalos_thissnap] = f_in["Ly"][halo_indices]
    forest_halos["Spinz"][current_offset:current_offset+NHalos_thissnap] = f_in["Lz"][halo_indices]
    
    forest_halos["MostBoundID"] [current_offset:current_offset+NHalos_thissnap]= f_in["ID"][halo_indices] ##

    forest_halos["SnapNum"][current_offset:current_offset+NHalos_thissnap]= snapnum 
    forest_halos["Filenr"][current_offset:current_offset+NHalos_thissnap] = filenr 
       
    forest_halos["SubHaloIndex"][current_offset:current_offset+NHalos_thissnap] = -1 ## 
    forest_halos["SubHalfMass"][current_offset:current_offset+NHalos_thissnap] = -1 ## 

    current_offset += NHalos_thissnap

    return forest_halos, current_offset 


def write_forest(fname_out, forest_halos):
   
     
    with open(fname_out, "ab") as f_out:

        f_out.write(forest_halos.tobytes())



def convert_binary_to_hdf5(fname_in, fname_out):
    """
    Converts a binary LHalo Tree file to HDF5 format.

    Parameters
    ----------

    fname_in, fname_out: String.
        Path to the input LHalo binary tree and the path 
        where the HDF5 tree will be saved.

    Returns
    ----------

    None
    """

    LHalo_Struct = get_LHalo_datastruct()

    with open(fname_in, "rb") as binary_file, \
         h5py.File(fname_out, "w") as hdf5_file:

        # First get header info from the binary file.
        NTrees = np.fromfile(binary_file, np.dtype(np.int32), 1)[0]
        NHalos = np.fromfile(binary_file, np.dtype(np.int32), 1)[0]
        NHalosPerTree = np.fromfile(binary_file,
                                    np.dtype((np.int32, NTrees)), 1)[0]

        print("For file {0} there are {1} trees with {2} total halos"
              .format(fname_in, NTrees, NHalos))

        # Write the header information to the HDF5 file.
        hdf5_file.create_group("Header")
        hdf5_file["Header"].attrs.create("Ntrees", NTrees, dtype=np.int32)
        hdf5_file["Header"].attrs.create("totNHalos", NHalos, dtype=np.int32)
        hdf5_file["Header"].attrs.create("TreeNHalos", NHalosPerTree,
                                         dtype=np.int32)

        # Now loop over each tree and write the information to the HDF5 file.
        for tree_idx in tqdm(range(NTrees)):
            binary_tree = np.fromfile(binary_file, LHalo_Struct,
                                      NHalosPerTree[tree_idx])

            tree_name = "tree_{0:03d}".format(tree_idx)
            hdf5_file.create_group(tree_name)

            for subgroup_name in LHalo_Struct.names:
                hdf5_file[tree_name][subgroup_name] = binary_tree[subgroup_name]

