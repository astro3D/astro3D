"""
Authors: Jacob Seiler, Manodeep Sinha
"""

#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py
from tqdm import tqdm
import os.path

import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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


def fix_nextprog(forest_halos):
    """
    Walks the descendants of a single forest to generate the `NextProgenitor`
    field.

    Parameters
    ----------

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        The halos within a single forest.
    
    Returns
    ----------

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        The forest halos with updated `NextProgenitor` field.
    """

    all_descendants = forest_halos["Descendant"][:]
    for ii, d in enumerate(all_descendants):
        if d == ii:
            continue
        curr = forest_halos["FirstProgenitor"][d]

        if curr == ii:
            continue
        while forest_halos["NextProgenitor"][curr] != -1:
            curr = forest_halos["NextProgenitor"][curr]

        assert(forest_halos["NextProgenitor"][curr] == -1)            
        forest_halos["NextProgenitor"][curr] = ii

    return forest_halos


def fix_flybys(forest_halos, NHalos_root):
    """
    Fixes flybys for a single forest. 

    Under the LHalo tree data structure, multiple FoFs at the root redshift are
    allowed IF AND ONLY IF all `FirstHaloInFOFgroup` values point to the same
    FoF. This is not enforced in the Treefrog data structure hence we must fix it
    here.

    We designate the most massive FoF halo at the root redshift to be the
    'True' FoF halo and update the Treefrog-equivalent field of
    `FirstHaloInFOFgroup` to point to this most massive halo. 

    Since we are adding an extra halo to the FoF group of the root snapshot, we
    also update the `NextHaloInFOFgroup` field. 

    ..note::
        We pass all halos within the forest to this function but only those at
        the root snapshot are altered. 

    Parameters
    ----------

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        The halos within a single forest.

    NHalos_root: Integer.
        The number of halos at the root snapshot.

    Returns
    ----------

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        The forest halos with updated `FirstHaloInFOFgroup` and
        `NextHaloInFOFgroup` fields. 
    """

    # Since we're at the root snapshot, the indexing will start from 0.
    root_halo_inds = np.arange(NHalos_root)
    root_halos = forest_halos[root_halo_inds]

    # If there is only one FoF Halo, no changes need to be made.
    if len(root_halo_inds) == 1:
        return forest_halos 

    # Find the 'true' FoF Halo.
    max_fof_mass_idx = np.argmax(forest_halos["Mvir"][root_halo_inds])

    # Find those halos whose `FirstHaloInFOFgroup` is incorrect
    flyby_ind = np.where(forest_halos["FirstHaloInFOFgroup"][root_halo_inds] 
                         != max_fof_mass_idx)[0]
    global_flyby_ind = root_halo_inds[flyby_ind]

    # Update them and flip the `MostBoundID` to flag the flyby.
    forest_halos["FirstHaloInFOFgroup"][global_flyby_ind] = max_fof_mass_idx 
    forest_halos["MostBoundID"][global_flyby_ind] *= -1 

    # Update the `NextHaloInFOFgroup` chain the account for the new FoFs.
    offset = 0
    forest_halos = fix_nextsubhalo(forest_halos, [max_fof_mass_idx], offset, 
                                   NHalos_root)

    return forest_halos 


def fix_nextsubhalo(forest_halos, fof_groups, offset, NHalos):
    """
    Fixes the `NextHaloInFOFgroup` field for a single forest at a single
    snapshot. 

    ..note::
        We pass all halos within the forest to this function but only alter
        those within a single snapshot. 

    Parameters
    ----------

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        The halos within a single forest.

    fof_groups: List of integers. 
        The FoF IDs for the halos being updated.  Necessary as we pass the
        entire forest. 

    offset: Integer.
        The (global) offset for the halos within the snapshot we're updating. 

    NHalos: Integer.
        The number of halos being updated.

    Returns
    ----------

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        The forest halos with updated `NextHaloInFOFgroup` field.        
    """

    # Every FoF group will point to a single halo, so loop over the FoF groups.
    for fof in fof_groups:

        # Find those halos within the snapshot we're altering in this FoF group.
        # We search only over the snapshot (offset -> offset+NHalos) to
        # increase efficiency for very large trees. 
        #print("FoF {0}".format(fof))
        halos_in_fof = np.where(forest_halos["FirstHaloInFOFgroup"][offset:offset+NHalos] == fof)[0]
        halos_in_fof_global_inds = halos_in_fof + offset

        '''
        print("Offset {0}".format(offset))
        print("NHalos {0}".format(NHalos))
        print("halos_in_fof {0}".format(halos_in_fof))
        print("halos_in_fof_global {0}".format(halos_in_fof_global_inds))
        '''
        # The first halo will point to index 1 and so on. 
        nexthalo = np.arange(offset+min(halos_in_fof)+1,
                             offset+min(halos_in_fof)+len(halos_in_fof)+1) 
        forest_halos["NextHaloInFOFgroup"][halos_in_fof_global_inds] = nexthalo
        # The final halo terminates with -1.
        forest_halos["NextHaloInFOFgroup"][halos_in_fof_global_inds[-1]] = -1 

        '''
        if fof == 547:
            print("Offset {0}".format(offset))
            print("NHalos {0}".format(NHalos))
            print("halos_in_fof {0}".format(halos_in_fof))
            print("halos_in_fof_global {0}".format(halos_in_fof_global_inds))
            print(halos_in_fof_global_inds[-1])            
            print(forest_halos["NextHaloInFOFgroup"][halos_in_fof_global_inds])
            print(forest_halos["NextHaloInFOFgroup"][halos_in_fof_global_inds[-1]])
        '''
    return forest_halos



def treefrog_to_lhalo(fname_in, fname_out, haloID_field="ID", 
                      forestID_field="ForestID"): 
    """
    Takes the Treefrog trees that have had their IDs corrected to be in LHalo
    format and saves them in LHalo binary format.

    The data-structure of the Treefrog trees is assumed to be HDF5 File ->
    Snapshots -> Halo Properties at each snapshot.

    ..note::
        We require the input trees to be sorted via the forest ID 
        (`forestID_field`) and suggest to also sub-sort on hostHaloID and mass.
        Sorting can be done using the `forest_sorter()` function.

        We also require the input trees to have IDs that are LHalo compatible;
        that is, they are forest local.

    ..note::
        The default parameters are chosen to match the ASTRO3D Genesis trees as
        produced by VELOCIraptor + Treefrog.
 
    Parameters
    ----------

    fname_in, fname_out: String.
        Path to the input HDF5 VELOCIraptor + treefrog trees and the path 
        where the LHalo binary file will be saved. 

    haloID_field: String. Default: 'ID'.
        Field name within the HDF5 file that corresponds to the unique halo ID.

    forestID_field: String. Default: 'ForestID'.
        Field name within the HDF5 file that corresponds to forest ID. 

    Returns
    ----------

    None.
    """

    if rank == 0:
        print("")
        print("=================================")
        print("Going through the LHalo indices corrected Treefrog trees and " 
              "saving in LHalo binary format.") 
        print("Input Trees: {0}".format(fname_in))
        print("Output LHalo ID Trees: {0}".format(fname_out))
        print("ForestID Field Name: {0}".format(forestID_field))
        print("Number of Processors: {0}".format(size))
        print("=================================")
        print("")
   
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

        total_forests_to_process = np.unique(f_in[last_snap_key][forestID_field][:])
        total_forests_to_process = total_forests_to_process[0:100]

        # If we're running in parallel, determine what forest IDs each
        # processor is handling.
        if size > 1: 
            forests_to_process = determine_forests(NHalos_forest,
                                                   total_forests_to_process) 
        
        filenr = rank

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
        print("Rank {0} writing {1} forests containing a total of {2} halos."\
              .format(rank, len(forests_to_process), totNHalos))
        my_fname_out = "{0}.{1}".format(fname_out, rank)
        write_header(my_fname_out, len(forests_to_process), totNHalos,
                     global_halos_per_forest)

        # Now for each forest we want to populate the LHalos forest struct, fix
        # any IDs (e.g., flybys) and then write them out.
        with open(my_fname_out, "ab") as f_out:
            for count, forestID in enumerate(forests_to_process):
                if count % 1000 == 0:
                    print("Rank {0} processed {1} Forests.".format(rank,
                                                                   count))
                NHalos = sum(NHalos_forest[forestID].values())

                forest_halos = np.zeros(NHalos, dtype=LHalo_Desc)
                forest_halos = populate_forest(f_in, forest_halos, Snap_Keys,
                                               Snap_Nums, forestID, NHalos_forest,
                                               filenr) 
       
                #print("Forest fully populated, now fixing up indices.")
                NHalos_root = NHalos_forest[forestID][last_snap_key]
                forest_halos = fix_flybys(forest_halos, NHalos_root)

                forest_halos = fix_nextprog(forest_halos)

                # The VELOCIraptor + Treefrog trees point to themselves when
                # they terminate.  However LHalo Trees requires these to be -1,
                # so find the instances where `NextProgenitor` and
                # `FirstProgenitor` point to themselves and adjust them to -1.
                w = np.arange(NHalos)
                NextProg_tofix = [x for x in w if x == forest_halos["NextProgenitor"][x]]
                FirstProg_tofix = [x for x in w if x == forest_halos["FirstProgenitor"][x]]

                forest_halos["NextProgenitor"][NextProg_tofix] = -1
                forest_halos["FirstProgenitor"][FirstProg_tofix] = -1

                # All done! Append to the file.
                forest_halos.tofile(f_out)

    print("Rank {0} has finished writing out {0} Forests to "
          "{1}".format(len(forests_to_process), my_fname_out))        


def determine_forests(NHalos_forest, all_forests):
    """
    Load balances the number of halos across processors.

    ..note::
        Since we do not split trees across processors, this function will
        result in processors having an unequal number of trees (but similar
        total number of halos). 
 
    Parameters
    ----------

    NHalos_forest: Nested Dictionary.
        Nested dictionary that contains the number of halos for each Forest at
        each snapshot.  Outer-key is the ForestID and inner-key is the snapshot
        key.

    all_forests: List of integers.
        List of Forest IDs to be processed across all files. 

    Returns
    ----------

    my_forest_assignment: List of integers.
        Rank-unique list of forest IDs to be processed by this rank.  
    """

    # We perform the operation on the root process and send the results out.
    if rank == 0:

        # First need to know how many halos we have across all forests.
        NHalos_total = 0
        for count, forestID in enumerate(all_forests):
            NHalos_total += sum(NHalos_forest[forestID].values())

        NHalo_target = NHalos_total / size  # Number of halos for each proc.

        # Create a nested list to hold the forest assignment for each processor.
        forest_assignment = []   

        # Start the assignment at the last rank and move down.  That way rank 0
        # will be last and the other processors can start their calculation.
        rank_count = size - 1
        halos_assigned = 0

        for count, forestID in enumerate(all_forests):
            # Loop over the forests until we have enough halos.
            halos_assigned += sum(NHalos_forest[forestID].values())
            forest_assignment.append(forestID)

            if halos_assigned > NHalo_target:
                print("Rank {0} has been assigned {1} forests with {2} total "
                      "halos.".format(rank_count,
                                      len(forest_assignment[rank_count]),
                                      halos_assigned))

                # Pass the assigned forest to the correct processor. 
                comm.send(forest_assignment, dest=rank_count, tag=1)

                # Then reset everything.
                rank_count -= 1
                halos_assigned = 0
                forest_assignment = []

        my_forest_assignment = forest_assignment
    else: 
        # All other ranks wait to receive their assignments.
        my_forest_assignment = comm.recv(source=0, tag=1)

    print("I am rank {0} and I have received {1} " 
          "forests.".format(rank, len(my_forest_assignment)))

    return my_forest_assignment 


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
                    NHalos_forest, filenr): 
    """
    Takes an empty `forest_halos` structure and fills it with the halos
    corresponding to the passed `forestID`.
 
    Parameters
    ----------

    f_in: Open HDF5 file. 
        The open HDF5 tree file we're reading from. 

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        Data structure to hold all the halos in this forest.  Has been
        initalized to zeros with enough length to hold all halos within this
        forest. 

    Snap_Keys: List of strings.
        List of keys that correspond to the fields containing the snapshot
        data.

    Snap_Nums: Dictionary of integers keyed by `Snap_Keys`.
        Snapshot number of each snapshot key.

    forestID: Integer.
        The forest ID we're populating the halos for.

    NHalos_forest: Nested Dictionary.
        Nested dictionary that contains the number of halos for each Forest at
        each snapshot.  Outer-key is the ForestID and inner-key is the snapshot
        key.
    
    filenr: Integer.
        The output file number this tree will be written to.

    Returns
    ----------

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        The filled data structure with the halos for the forest. 
    """
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

        # Then go to the HDF5 file and grab all the required properties.
        forest_halos, halos_offset = fill_LHalo_properties(f_in[snap_key], 
                                                           forest_halos, 
                                                           halos_forest_inds, 
                                                           halos_offset,
                                                           Snap_Nums[snap_key],
                                                           filenr)

    return forest_halos


def fill_LHalo_properties(f_in, forest_halos, halo_indices, current_offset,
                          snap_num, filenr):
    """
    Grabs all the properties from the input HDF5 file for the current snapshot. 

    ..note::
        The passed parameter `f_in` should correspond to the **snapshot group**
        that contains the halo properties datasets.  E.g., if we are filling
        the halos for Snapshot 43 with key `Snap_043`, this function should be
        called with `f_in[Snap_043]`.
 
    Parameters
    ----------

    f_in: Open HDF5 file. 
        The **snapshot level** HDF5 group that we're reading from. 

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        Data structure to hold all the halos for this forest.  

        ..note::
            We pass the entire forest array here but only update a slice of it. 

    halo_indices: List of integers.
        List of indices that correspond to the halos we're filling for this
        snapshot.

    current_offset: Integer.
        The index into the `forest_halo` structure where this snapshot starts.
   
    snap_num: Integer.
        The snapshot number we're filling for.
 
    filenr: Integer.
        The output file number this tree will be written to.

    Returns
    ----------

    forest_halos: `~np.ndarray` with data structure defined by
                  `get_LHalo_datastruct()`
        The filled data structure with the halos for the forest. 
    """

    NHalos_thissnap = len(halo_indices)
    
    forest_halos["Descendant"][current_offset:current_offset+NHalos_thissnap] = f_in["Head"][halo_indices]
    forest_halos["FirstProgenitor"][current_offset:current_offset+NHalos_thissnap]  = f_in["Tail"][halo_indices]
    
    # Initialize `NextProgenitor` with -1 and then fix it later with function
    # `fix_nextprog()`.
    forest_halos["NextProgenitor"][current_offset:current_offset+NHalos_thissnap] = -1

    # `FirstHaloInFOFgroup` points to itself for the main FoF halo. LHalo Tree
    # requires -1 but we will fix this later. 
    forest_halos["FirstHaloInFOFgroup"][current_offset:current_offset+NHalos_thissnap] = f_in["hostHaloID"][halo_indices]

    # First find out what the FoF groups are.  Then go through the FoF groups
    # and update the `NextHaloInFOFgroup` pointer.
    forest_halos["NextHaloInFOFgroup"][current_offset:current_offset+NHalos_thissnap] = -1
    all_hosthalo_inds = f_in["hostHaloID"][halo_indices]
    fof_groups = np.unique(all_hosthalo_inds)
    forest_halos = fix_nextsubhalo(forest_halos, fof_groups, current_offset,
                                   NHalos_thissnap)

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
    
    forest_halos["MostBoundID"][current_offset:current_offset+NHalos_thissnap]= f_in["oldIDs"][halo_indices]

    forest_halos["SnapNum"][current_offset:current_offset+NHalos_thissnap]= snap_num 
    forest_halos["Filenr"][current_offset:current_offset+NHalos_thissnap] = filenr 
       
    forest_halos["SubHaloIndex"][current_offset:current_offset+NHalos_thissnap] = -1 ## 
    forest_halos["SubHalfMass"][current_offset:current_offset+NHalos_thissnap] = -1 ## 

    current_offset += NHalos_thissnap

    return forest_halos, current_offset 


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

    None.
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

