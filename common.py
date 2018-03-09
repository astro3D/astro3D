#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import h5py

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
