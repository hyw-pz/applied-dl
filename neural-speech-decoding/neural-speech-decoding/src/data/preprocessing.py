"""
preprocessing.py
----------------
HDF5 data loading, file management, and neural signal normalisation.
Supports both single-file and multi-session loading with Drive→local caching.
"""

import glob
import os
import shutil

import h5py
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Phoneme vocabulary
# ─────────────────────────────────────────────────────────────────────────────

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B',  'CH', 'D',  'DH',
    'EH', 'ER', 'EY', 'F',  'G',  'HH', 'IH', 'IY', 'JH', 'K',
    'L',  'M',  'N',  'NG', 'OW', 'OY', 'P',  'R',  'S',  'SH',
    'T',  'TH', 'UH', 'UW', 'V',  'W',  'Y',  'Z',  'ZH', '|',
]
NUM_PHONEMES = len(LOGIT_TO_PHONEME)  # 41


# ─────────────────────────────────────────────────────────────────────────────
# Single-file loader
# ─────────────────────────────────────────────────────────────────────────────

def load_h5py_file(file_path: str) -> dict:
    """
    Load one .hdf5 session file.

    Neural features are stored as-is (T, 512); clipping/padding
    happens lazily in the Dataset at __getitem__ time.

    Returns
    -------
    dict with keys:
        neural_features : list of np.ndarray  (T_i, 512)
        n_time_steps    : list of int
        seq_class_ids   : list of np.ndarray  (max_seq,)
        seq_len         : list of int
        transcriptions  : list[bytes | None]
        sentence_label  : list[str | None]
        session         : list[str]
        block_num       : list[int]
        trial_num       : list[int]
    """
    data = {k: [] for k in (
        'neural_features', 'n_time_steps', 'seq_class_ids',
        'seq_len', 'transcriptions', 'sentence_label',
        'session', 'block_num', 'trial_num',
    )}

    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            g = f[key]

            neural_features = g['input_features'][:].astype(np.float32)
            n_time_steps    = int(g.attrs['n_time_steps'])
            seq_class_ids   = (g['seq_class_ids'][:].astype(np.int64).flatten()
                               if 'seq_class_ids' in g else None)
            seq_len         = int(g.attrs['seq_len']) if 'seq_len' in g.attrs else None
            transcription   = g['transcription'][:] if 'transcription' in g else None
            sentence_label  = (g.attrs['sentence_label'][:]
                               if 'sentence_label' in g.attrs else None)

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(g.attrs['session'])
            data['block_num'].append(g.attrs['block_num'])
            data['trial_num'].append(g.attrs['trial_num'])

    print(f'Loaded {len(data["neural_features"])} trials from {file_path}')
    return data


def _merge_dicts(dicts: list) -> dict:
    """Concatenate a list of same-keyed data dicts into one."""
    merged = {k: [] for k in dicts[0]}
    for d in dicts:
        for k in merged:
            merged[k].extend(d[k])
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Multi-session loader with Drive→local caching
# ─────────────────────────────────────────────────────────────────────────────

def load_all_files(
    drive_dir: str,
    local_dir: str,
    split: str,
    max_files: int = None,
) -> dict:
    """
    Discover all ``data_{split}.hdf5`` files in *drive_dir*, copy them to
    *local_dir* (skipping already-copied files), then load and merge.

    If enough local copies already exist the Drive is not queried at all.

    Parameters
    ----------
    drive_dir  : root directory to search on Google Drive (or any mount)
    local_dir  : local directory used as a cache
    split      : one of ``'train'``, ``'val'``, ``'test'``
    max_files  : cap on the number of session files (``None`` = all)

    Returns
    -------
    Merged data dict (same structure as :func:`load_h5py_file`)
    """
    local_pattern = os.path.join(local_dir, '**', f'data_{split}.hdf5')
    local_paths   = sorted(glob.glob(local_pattern, recursive=True))

    skip_drive = False
    if local_paths:
        if max_files is None or len(local_paths) >= max_files:
            skip_drive = True
            if max_files is not None:
                local_paths = local_paths[:max_files]

    if skip_drive:
        print(f"Found {len(local_paths)} local files for '{split}'. Skipping Drive.")
    else:
        drive_pattern = os.path.join(drive_dir, '**', f'data_{split}.hdf5')
        drive_paths   = sorted(glob.glob(drive_pattern, recursive=True))
        if not drive_paths:
            raise FileNotFoundError(
                f"No 'data_{split}.hdf5' found in {drive_dir}")
        if max_files is not None:
            drive_paths = drive_paths[:max_files]

        print(f"Copying {len(drive_paths)} files for '{split}' to local disk…")
        local_paths = []
        for dp in drive_paths:
            rel = os.path.relpath(dp, drive_dir)
            lp  = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            if not os.path.exists(lp):
                shutil.copy2(dp, lp)
            local_paths.append(lp)

    print(f"Loading and merging {len(local_paths)} session files…")
    return _merge_dicts([load_h5py_file(p) for p in local_paths])


# ─────────────────────────────────────────────────────────────────────────────
# Signal normalisation (used outside the Dataset when needed)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_neural_data(neural_data: np.ndarray) -> np.ndarray:
    """
    Z-score normalise and reshape a batch of neural arrays for Conv2d input.

    Parameters
    ----------
    neural_data : np.ndarray  shape (N, T, 512)

    Returns
    -------
    np.ndarray  shape (N, 1, 512, T)
    """
    mean = neural_data.mean(axis=(0, 1), keepdims=True)
    std  = neural_data.std(axis=(0, 1),  keepdims=True) + 1e-8
    neural_data = (neural_data - mean) / std
    neural_data = neural_data.transpose(0, 2, 1)     # (N, 512, T)
    neural_data = np.expand_dims(neural_data, 1)     # (N, 1, 512, T)
    return neural_data


def decode_phoneme_ids(seq_ids: list) -> str:
    """Convert a list of phoneme integer IDs to an ARPAbet string."""
    return " ".join(
        LOGIT_TO_PHONEME[p]
        for p in seq_ids
        if p != 0 and p < NUM_PHONEMES
    )


def decode_text_label(label) -> str:
    """Decode bytes or str sentence label to a plain string."""
    if isinstance(label, bytes):
        return label.decode('utf-8')
    return str(label)
