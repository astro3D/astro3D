#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py
from tqdm import tqdm
from optparse import OptionParser

import time

def parse_inputs():
    """

    Parses the command line input arguments.  If there has not been an input or output file name specified a RuntimeError will be raised. 
    
    Parameters
    ----------

    None.   
 
    Returns
    ----------

    opt: optparse.Values.  Required.  
        Values from the OptionParser package.  Values are accessed through ``opt.Value`` and cast into a dictionary using ``vars(opt)`` 
    """

    parser = OptionParser()

    parser.add_option("-f", "--fname_in", dest="fname_in", help="Path to the input HDF5 data file. Required.")
    parser.add_option("-o", "--fname_out", dest="fname_out", help="Path to the output HDF5 data file. Required.")
    parser.add_option("-s", "--sort_id", dest="sort_id", help="Field name for the key we are sorting on. Default: ForestID.", default = "ForestID")
    parser.add_option("-m", "--mass_def", dest="sort_mass", help="Field name for the mass key we are sorting on. Default: Mass_200mean.", default = "Mass_200mean")
    parser.add_option("-i", "--HaloID", dest="halo_id", help="Field name for halo ID. Default: ID.", default = "ID")
    parser.add_option("-p", "--ID_fields", dest="ID_fields", help="Field names for those that contain IDs.  Default: ('ID', 'Tail', 'Head', 'NextSubHalo', 'Dummy1', 'Dumm2').", default = ("ID", "Tail", "Head", "NextSubHalo", "Dummy", "Dummy")) 
    parser.add_option("-x", "--index_mult_factor", dest="index_mult_factor", help="Conversion factor to go from a unique, per-snapshot halo index to a temporally unique haloID.  Default: 1e12.", default = 1e12)

    (opt, args) = parser.parse_args()

    if (opt.fname_in == None or opt.fname_out == None): # If the required parameters have not been supplied, throw an exception.
        parser.print_help()
        raise RuntimeError

    # Print some useful startup info. #
    print("")
    print("The HaloID field for each halo is '{0}'.".format(opt.halo_id)) 
    print("Sorting on the '{0}' field.".format(opt.sort_id))
    print("Sub-Sorting on the '{0}' field.".format(opt.sort_mass))
    print("")

    return opt

def get_sort_indices(file_in, snap_key, opt):
    """

    Gets the indices that will sort the HDF5 file using the two keys provided by the user in opt.
    The first key specifies the order of the "outer-sort" with the second key specifying the order 
    of the "inner-sort" within each group sorted by the first key. 

    Example:
        Outer-sort uses ForestID and inner-sort used Mass_200mean.
        ForestID = [1, 4, 39, 1, 1, 4]
        Mass_200mean = [4e9, 10e10, 8e8, 7e9, 3e11, 5e6]
        
        Then the indices would be [0, 3, 4, 5, 1, 2]

    Parameters
    ----------

    file_in: HDF5 file.  Required.
        Open HDF5 file that we are sorting for. The data structure is assumed to be 
        HDF5_File -> List of Snapshot Numbers -> Halo properties/pointers.

    snap_key: String.  Required.
        The field name for the snapshot we are accessing.

    opt: Dictionary.  Required.
        Dictionary containing the option parameters specified at runtime.  Used to specify the field names we are sorting on. 
        
    Returns
    ----------

    indices: numpy-array.  Required.
        Array containing the indices that sorts the keys for the specified dataset. 

    """ 
    indices = np.lexsort((file_in[snap_key][opt["sort_mass"]], file_in[snap_key][opt["sort_id"]])) # Sorts via the ID then sub-sorts via the mass

    return indices

def snap_key_to_snapnum(snap_key):
    """

    Given the name of a snapshot key, we wish to find the snapshot number associated with this key.
    This is necessary because the 0th snapshot key may not be snapshot 000 and there could be missing snapshots (e.g., snapshot 39 is followed by snapshot 41).

    This function takes the key and searches backwards for a group of digits that identify the snapshot number. 
    The function will only consider numbers that are clustered together, starting at the end of the snapshot key.  If there are numbers outside of this cluster they will be disregarded and a warning raised.
    For example, if the key is "Snap1_030", the function will return 30 and issue a warning that there were digits ignored. 
    
    Parameters
    ----------

    snap_key: String.  Required. 
        The name of the snapshot key.    
    
    Returns
    ----------

    snapnum: integer.  Required.
        The snapshot number that corresponds to the snapshot key. 

    """

    snapnum = ""
    reached_numbers = False

    for letter in reversed(snap_key): # Go backwards through the key 
        if (letter.isdigit() == True): # When a number is found,            
            snapnum = "{0}{1}".format(snapnum, letter) # Concatenate that number with the others.
            reached_numbers = True # Flag that we have encountered a cluster of numbers.

        if (letter.isdigit() == False): # When we eventually reach a letter that is not a number,
            reached_numbers = False # Turn the flag off.

        if (letter.isdigit() == True and reached_numbers == False): # But if we keep going back and encounter a number again, raise a Warning.
            Warning("For Snapshot key '{0}' there were numbers that were not clustered together at the end of the key.\nWe assume the snapshot number corresponding to this key is {1}; please check that this is correct.".format(snap_key, int(snapnum))) 
    
    snapnum = snapnum[::-1] # We searched backwards so flip the string around.

    return int(snapnum) # Cast as integer before returning.

def index_to_temporalID(index, snapnum, index_mult_factor):
    """

    Given a haloID local to a snapshot with number ``snapnum``, this function returns the ID that accounts for the snapshot number. 
    
    Parameters
    ----------

    index: array-like of integers, or integer. Required.
        Array or single value that describes the snapshot-local haloID.

    snapnum: integer.  Required
        Snapshot that the halo/s are/is located at.

    index_mult_factor: integer. Required
        Factor to convert a the snapshot-unique halo index to a temporally-unique halo ID.
 
    Returns
    ----------

    index: array-like of integers, or integer. Required.    
        Array or single value that contains the temporally unique haloID.         
    """

    temporalID = snapnum*int(index_mult_factor) + index + 1

    return temporalID


def get_snapkeys_and_nums(file_keys):
    """

    Grabs the names of the snapshot keys and associated snapshot
    numbers from a given set of keys.
    
    We assume that the snapshot data keys are named to include the word
    "snap" (case insensitive). We also assume that the snapshot number
    for each snapshot key will be in a single cluster towards the end
    of the key. If this is not the case we issue a warning showing what
    we believe to be the corresponding snapshot number.

    Parameters
    ----------

    file_keys: Keys. 
        Keys from a given file or dataset.
         
    Returns
    ----------

    Snap_Keys: List of strings. 
        Names of the snapshot keys within the passed keys.

    Snap_Num: Dictionary of integers keyed by Snap_Keys.
        Snapshot number of each snapshot key. 

    """

    Snap_Keys = [key for key in file_keys if (("SNAP" in key.upper()) == True)] 
    Snap_Nums = dict() 
    for key in Snap_Keys: 
        Snap_Nums[key] = snap_key_to_snapnum(key) 

    return Snap_Keys, Snap_Nums 

def temporalID_to_snapnum(temporalID, index_mult_factor): 
    """

    Given a temporalID, this function returns the snapshot number that corresponds to the ID. 
    
    Parameters
    ----------

    ID: array-like of integers, or integer. Required.
        Array or single value that describes the temporalID/s. 

    index_mult_factor: integer. Requied.
        Factor to convert a the snapshot-unique halo index to a temporally-unique halo ID.
         
    Returns
    ----------

    snapnum: array-like of integers, or integer. Required.    
        Array or single value that contains the snapshot number corresponding to the temporal ID. 
    """

    if (isinstance(temporalID, list)) or (type(temporalID).__module__ == np.__name__):
        snapnum = (np.subtract(temporalID, 1) / index_mult_factor).astype(int)
    else:        
        snapnum = int((temporalID - 1) / index_mult_factor)

    return snapnum
    
def copy_group(file_in, file_out, key, opt):
    """

    Copies a group (and it's nested data-structure) within a HDF5 group into a new HDF5 file with the same data-structure.
 
    Parameters
    ----------

    file_in, file_out: Open HDF5 files.  Required.
        HDF5 files for the data being copied (file_in) and the file the data is being copied to (file_out). 

    key: String.  Required.
        Name of the HDF5 group being copied. 

    opt: Dictionary.  Required.
        Dictionary containing the option parameters specified at runtime.  Used to specify the field names we are sorting on. 
     
    Returns
    ----------

    None. 

    """

    group_path = file_in[key].parent.name # Get the name of the group path in the input file.
    group_id = file_out.require_group(group_path) # Create the group (and relevant sub-structure) if necessary.
    name = "{0}".format(key) # Name the group.
    file_in.copy(name, group_id, name = key) # Copy over the data.

def sort_and_write_file(opt):
    """

    Using the options specified by the command line, sorts the HDF5
    file by the specified ID field and then sub-sorts by the specified
    mass field.

    The output file will be saved in this sorted order. 
 
    Parameters
    ----------

    opt: Dictionary.  Required.
        Contains the runtime variables such as input/output file names
        and fields required for sorting.
        For full contents of the dictionary refer to ``parse_inputs``.  
     
    Returns
    ----------

    None. 

    """

    with h5py.File(opt["fname_in"], "r") as f_in, h5py.File(opt["fname_out"], "w") as f_out:

        Snap_Keys, Snap_Nums = get_snapkeys_and_nums(f_in.keys())
       
        ID_maps = dict() 
        snapshot_indices = dict() 

        print("")
        print("Generating the dictionary to map the oldIDs to the newIDs.")

        start_time = time.time() 
        for snap_key in tqdm(Snap_Keys):
            if len(f_in[snap_key][opt["halo_id"]]) == 0:  # Skip empty snapshots. 
                continue
 
            indices = get_sort_indices(f_in, snap_key, opt)  # Indices that will sort the snapshot
                                                             # dataset according to specified fields. 
            old_haloIDs = f_in[snap_key][opt["halo_id"]][:]
            old_haloIDs_sorted = old_haloIDs[indices] 
            new_haloIDs = index_to_temporalID(np.arange(len(indices)), Snap_Nums[snap_key],
                                              opt["index_mult_factor"])
                                              # HaloIDs are given by their snapshot-local index.
 
            oldIDs_to_newIDs = dict(zip(old_haloIDs_sorted, new_haloIDs)) 
            
            snapshot_indices[snap_key] = indices  # Dictionary keyed by the snapshot field.
            ID_maps[Snap_Nums[snap_key]] = oldIDs_to_newIDs  # Nested dictionary keyed by snapnum. 


        end_time = time.time() 
        print("Creation of dictionary map took {0:3f} seconds".format(end_time - start_time))
        print("Done!")
        print("")

        # At this point we have the dictionaries that map the oldIDs to the newIDs in addition to
        # the indices that control the sorting of the array.  We now loop through all the fields 
        # within each halo halo within each snapshot and if the field contains a haloID we update
        # it.  While going through each field, we will then write out the data into a new HDF5 file
        # in the order specified by indices.

        print("")
        print("Now writing out the snapshots in the sorted order.")
        start_time = time.time()
 
        for key in f_in.keys():  # Loop through snapshots.
            copy_group(f_in, f_out, key, opt)
            for field in f_in[key]:  # Then through each field 

                # Only need to do the sorting for keys that are snapshots with Halos. 
                try: 
                    NHalos = len(f_in[key][opt["halo_id"]])
                    if (NHalos == 0):
                        continue
                except KeyError:  # Some keys (e.g., 'Header') don't contain halos. 
                    continue

                
                if field in opt["ID_fields"]:  # If we need to update the ID for this field.
                    newID = np.empty((NHalos)) 
                    for idx in range(NHalos):  # Loop through each halo.
                        oldID = f_in[key][field][idx] 
                        snapnum = temporalID_to_snapnum(oldID, opt["index_mult_factor"])
                        newID[idx] = int(ID_maps[snapnum][oldID])
                    to_write = newID
                else:
                    to_write = f_in[key][field][:]               
 
                f_out[key][field][:] = to_write[snapshot_indices[key]] # Then reorder output.

        end_time = time.time() 
        print("Writing of snapshots took {0:3f} seconds".format(end_time - start_time))
        print("Done!")        
        print("")        

if __name__ == '__main__':

    opt = parse_inputs()
    sort_and_write_file(vars(opt))

