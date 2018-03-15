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
    """

    snapnum = ""
    reached_numbers = False

    for letter in reversed(snap_key):  # Go backwards through the key.
        if letter.isdigit():
            # When a number is found, we concatenate it with the others and
            # flag that we have encountered a cluster of numbers.
            snapnum = "{0}{1}".format(snapnum, letter)
            reached_numbers = True

        else:
            # When we reach something that's not a number, turn flag off.
            reached_numbers = False

        # Now if we reach a number outside of the cluster at the end of the
        # key, raise a warning that there were numbers ignored.
        if letter.isdigit() and not reached_numbers:
            Warning("For Snapshot key '{0}' there were numbers that were not "
                    "clustered together at the end of the key.\nWe assume the "
                    "snapshot number corresponding to this key is {1}; please "
                    "check that this is correct."
                    .format(snap_key, int(snapnum)))

    snapnum = snapnum[::-1]  # We searched backwards so flip the string around.

    return int(snapnum)  # Cast as integer before returning.


def index_to_temporalID(index, snapnum, index_mult_factor):
    """
    Takes snapshot-local halo index and converts into temporally unique ID.

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
    """

    snapnum = int((temporalID - 1) / index_mult_factor)

    return snapnum


def copy_group(file_in, file_out, key, opt):
    """
    Copies HDF5 group into a new HDF5 file (with same data-structure).

    Parameters
    ----------

    file_in, file_out: Open HDF5 files.  Required.
        HDF5 files for the data being copied (file_in) and the file the
        data is being copied to (file_out).

    key: String.  Required.
        Name of the HDF5 group being copied.

    opt: Dictionary.  Required.
        Dictionary containing the option parameters specified at runtime.
        Used to specify the field names we are sorting on.

    Returns
    ----------

    None.
    """

    group_path = file_in[key].parent.name  # Name of the group path.
    group_id = file_out.require_group(group_path)  # Create the group.
    name = "{0}".format(key)  # Name the group.
    file_in.copy(name, group_id, name=key)  # Copy over the data.
