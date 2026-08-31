"""Load MATLAB .mat files of any version behind one interface.

Files under 2 GB are saved in MATLAB's classic format (scipy reads them); larger
files must be saved as v7.3, which is HDF5 in disguise and needs h5py. The two
readers disagree on two things, so this module hides both:

  - axis order: HDF5 reads MATLAB's column-major data reversed, so a MATLAB
    (77, 246, 36000) array arrives as (36000, 246, 77). We transpose it back.
  - strings: v7.3 stores them as references into a '#refs#' group holding uint16
    character codes. We dereference and decode them to plain str.
"""

import numpy as np
import scipy.io
import h5py


def is_v73(path):
    """True if the file is MATLAB v7.3 (HDF5-based).

    Uses h5py's own detector rather than sniffing the header text, which can
    false-positive on classic files that merely mention "7.3" somewhere.
    """
    return h5py.is_hdf5(path)


def _h5_str(f, ref):
    """Decode one MATLAB string stored as an HDF5 reference."""
    return ''.join(chr(c) for c in np.asarray(f[ref]).flatten())


def _sp_str(x):
    """Decode one MATLAB string as scipy returns it (nested 1-element array)."""
    a = np.asarray(x).flatten()
    return str(a[0]) if a.size else ''


def peek(path):
    """Print each variable's shape and dtype without loading the data.

    Safe on multi-GB files: h5py reads lazily, so this only touches metadata.
    """
    print(f'{path}  ({"v7.3 / HDF5" if is_v73(path) else "classic"})')
    if is_v73(path):
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                if k.startswith('#'):
                    continue                      # '#refs#' is MATLAB bookkeeping
                d = f[k]
                shape = tuple(reversed(d.shape))  # report MATLAB's axis order
                print(f'  {k:12s} {str(shape):24s} {d.dtype}')
    else:
        raw = scipy.io.loadmat(path)
        for k, v in raw.items():
            if k.startswith('__'):
                continue
            print(f'  {k:12s} {str(v.shape):24s} {v.dtype}')


def load_mat(path, keys=None):
    """Load a .mat file into a plain dict, regardless of version.

    keys: load only these variables. Worth using on large files so you can read
    metadata (ids, labels) without pulling a multi-GB array into memory.

    Numeric arrays come back in MATLAB's axis order; string cell arrays come
    back as lists of str.
    """
    out = {}
    if is_v73(path):
        with h5py.File(path, 'r') as f:
            names = keys or [k for k in f.keys() if not k.startswith('#')]
            for k in names:
                d = f[k]
                if d.dtype == object:                        # cell array of strings
                    out[k] = [_h5_str(f, r) for r in np.asarray(d).flatten()]
                else:
                    out[k] = np.asarray(d).T                 # undo HDF5 axis reversal
    else:
        raw = scipy.io.loadmat(path)
        names = keys or [k for k in raw.keys() if not k.startswith('__')]
        for k in names:
            v = raw[k]
            if getattr(v, 'dtype', None) == object:          # cell array of strings
                out[k] = [_sp_str(x) for x in v.flatten()]
            else:
                out[k] = v
    return out
