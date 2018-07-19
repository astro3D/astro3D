#!/usr/bin/env python

from __future__ import print_function

def snap_key_to_snapnum(snap_key):
    """
    Given the name of a snapshot key, finds the associated snapshot number.

    This is necessary because the 0th snapshot key may not be snapshot 000 and
    there could be missing snapshots. This function searches backwards for a
    group of digits that identify the snapshot number.  If there are numbers
    outside of this cluster they will be disregarded and a warning raised.

    For example, if the key is "Snap1_030", the function will return 30 and
    issue a warning that there were digits ignored.

    Parameters
    ----------

    snap_key: String.  Required.
        The name of the snapshot key.

    Returns
    ----------

    snapnum: Integer.  Required.
        The snapshot number that corresponds to the snapshot key.

    Examples
    ----------

    >>> snap_key_to_snapnum('Snap_018')
    18

    >>> snap_key_to_snapnum('018_Snap')
    18

    >>> snap_key_to_snapnum('Sn3p_018')
    --WARNING--
    For Snapshot key 'Sn3p_018' there were numbers that were not \
clustered together at the end of the key.
    We assume the snapshot number corresponding to this key is 18; \
please check that this is correct.
    18
    """

    snapnum = ""
    reached_numbers = None

    for letter in reversed(snap_key):  # Go backwards through the key.
        if letter.isdigit():
            if reached_numbers == False and len(snapnum):
                print("--WARNING--")
                print("For Snapshot key '{0}' there were numbers that were not"
                      " clustered together at the end of the key.\nWe assume "
                      "the snapshot number corresponding to this key is {1}; "
                      "please check that this is correct."
                      .format(snap_key, int(snapnum[::-1])))
                break 
            # When a number is found, we concatenate it with the others and
            # flag that we have encountered a cluster of numbers.
            snapnum = "{0}{1}".format(snapnum, letter)
            reached_numbers = True 

        else:
            # When we reach something that's not a number, turn flag off.
            reached_numbers = False

    snapnum = snapnum[::-1]  # We searched backwards so flip the string around.

    return int(snapnum)  # Cast as integer before returning.


def index_to_temporalID(index, snapnum, index_mult_factor):
    """
    Takes snapshot-local halo index and converts into temporally unique ID.

    Note: IDs start counting at 1.  So the index 0 gets mapped to an ID of 1.

    Parameters
    ----------

    index: array-like of integers, or integer. Required.
        Array or single value that describes the snapshot-local haloID.

    snapnum: integer.  Required
        Snapshot that the halo/s are/is located at.

    index_mult_factor: integer. Required
        Factor to convert a the snapshot-unique halo index to a temporally
        unique halo ID.

    Returns
    ----------
    
    index: array-like of integers, or integer. Required.
        Array or single value that contains the temporally unique haloID.

    Examples
    ----------

    >>> index_to_temporalID(23, 18, 1e12)
    18000000000024

#    >>> test_list = [25, 100, 4]
#    >>> test_snapnums = [23, 12, 60]
#    >>> index_to_temporalID(test_list, test_snapnums, 1e12)
    [2300000026, 1200000101, 600000005] 
    """

    temporalID = snapnum*int(index_mult_factor) + index + 1

    return temporalID


def get_snapkeys_and_nums(file_keys):
    """
    Gets names of snapshot keys and snapshot numbers.

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

    Examples
    ----------

    """

    Snap_Keys = [key for key in file_keys if ("SNAP" in key.upper())]
    Snap_Nums = dict()
    for key in Snap_Keys:
        Snap_Nums[key] = snap_key_to_snapnum(key)

    return Snap_Keys, Snap_Nums


def temporalID_to_snapnum(temporalID, index_mult_factor):
    """
    Given a temporalID, return the corresponding snapshot number.

    Parameters
    ----------

    ID: array-like of integers, or integer. Required.
        Array or single value that describes the temporalID/s.

    index_mult_factor: integer. Required.
        Factor to convert to from temporally-unique halo ID to snap-shot unique
        halo index.

    Returns
    ----------

    snapnum: array-like of integers, or integer. Required.
        Array or single value that contains the snapshot number corresponding
        to the temporal ID.

    Examples
    ----------

    >>> temporalID_to_snapnum(-1, 1e12)
    0

    >>> temporalID_to_snapnum(18000000000001, 1e12)
    18

    >>> test_list = [18000000000001, 20000000000050, 134000000000005]
    >>> temporalID_to_snapnum(test_list, 1e12)
    array([ 18,  20, 134])


    >>> import numpy as np
    >>> test_array = np.array([20000000000050, 134000000000005])
    >>> temporalID_to_snapnum(test_array, 1e12)
    array([ 20, 134])

    """
    import numpy as np
    if isinstance(temporalID, list) or isinstance(temporalID, np.ndarray):
        snapnum = ((np.subtract(temporalID,1)) / index_mult_factor).astype(int)
    else:
        snapnum = int((temporalID - 1) / index_mult_factor)

    return snapnum


def copy_group(file_in, file_out, key):
    """
    Copies HDF5 group into a new HDF5 file (with same data-structure).

    Parameters
    ----------

    file_in, file_out: Open HDF5 files.  Required.
        HDF5 files for the data being copied (file_in) and the file the
        data is being copied to (file_out).

    key: String.  Required.
        Name of the HDF5 group being copied.

    Returns
    ----------

    None.
    """

    group_path = file_in[key].parent.name  # Name of the group path.
    group_id = file_out.require_group(group_path)  # Create the group.
    name = "{0}".format(key)  # Name the group.
    file_in.copy(name, group_id, name=key)  # Copy over the data.


def get_halos_per_forest(f_in, Snap_Keys, halo_id_name, forest_id_name):
    """
    Determines the number of halos in each forest.

    The resulting Dictionary is nested with the outer-key given by the ForestID
    and the inner-key given by the snapshot field name.

    E.g., If Forest 5 has 10 halos at Snapshot 20 and 100 at Snapshot 21 then,
        NHalos_Forest[5]['Snap_020'] = 10
        NHalos_Forest[5]['Snap_021'] = 100

    We also generate the offset for each Forest at each snapshot.  This is
    necessary because whilst Forest 5 may saved first at snapshot 20, it isn't
    necessarily saved first at snapshot 21.

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

    NHalos_Forest_Offset: Nested Dictionary. Required.
        Nested dictionary that contains the offset for each Forest at each
        snapshot. Outer-key is the ForestID and inner-key is the snapshot key.
        This is required because whilst the tree is sorted by ForestID, the
        relative position of the tree can change from snapshot to snapshot.
    """

    import numpy as np
    import time
    from tqdm import tqdm

    start_time = time.time()

    print("")
    print("Generating the dictionary for the number of halos in each forest "
          "at each snapshot.")

    NHalos_Forest = {}
    NHalos_Forest_Offset = {}

    for count, snap_key in enumerate(tqdm(Snap_Keys)):
        if len(f_in[snap_key][halo_id_name]) == 0:  # Skip empty snapshots.
            continue

        halos_counted = 0
        halo_forestids = f_in[snap_key][forest_id_name][:]

        # First get the number of halos in each forest then grab the indices
        # (i.e., the forestID as we start from 0) of the forests that have
        # halos.
        forests_binned = np.bincount(halo_forestids)
        forestIDs = np.nonzero(forests_binned)[0]

        for forest_id in forestIDs:
            this_snap_NHalos = forests_binned[forest_id]

            # The first time a forest appears it won't have a corresponding key
            # in the nested dictionary so create it if it's the case.
            try:
                NHalos_Forest[forest_id][snap_key] = this_snap_NHalos
                NHalos_Forest_Offset[forest_id][snap_key] = halos_counted
            except KeyError:
                NHalos_Forest[forest_id] = {snap_key: this_snap_NHalos}
                NHalos_Forest_Offset[forest_id] = {snap_key: halos_counted}

            halos_counted += this_snap_NHalos

    end_time = time.time()
    print("Creation of number of halos per forest took {0:3f} seconds."
          .format(end_time - start_time))
    print("")

    return NHalos_Forest, NHalos_Forest_Offset


if __name__ == "__main__":
    import doctest
    import numpy as np
    test_list = [25, 100, 4]
    test_snapnums = [23, 12, 60]
    print(index_to_temporalID(list(np.arange(0,10)), list(np.arange(0,10)), 1e12))
    print("QJTOETQ")
    doctest.testmod()
