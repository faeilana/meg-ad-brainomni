import numpy as np
import scipy.io
import scipy.signal
import torch

FS_ORIG = 600
FS_TARGET = 256
SEGMENT_SECS = 30


def preprocess_subject(mat_path):
    data = scipy.io.loadmat(mat_path)

    signal = data['trial'][0, 0].astype(np.float32)
    signal = scipy.signal.resample_poly(signal, up=32, down=75, axis=1).astype(np.float32)

    grad = data['grad']
    grad_labels = [str(grad[0,0]['label'][i, 0][0]) for i in range(grad[0,0]['label'].shape[0])]
    meg_labels  = [str(data['label'][i, 0][0]) for i in range(data['label'].shape[0])]
    idx = [grad_labels.index(lbl) for lbl in meg_labels]

    chanpos = grad[0,0]['chanpos'][idx].astype(np.float32)
    chanori = grad[0,0]['chanori'][idx].astype(np.float32)
    pos = np.concatenate([chanpos, chanori], axis=1)

    pos[:, :3] -= pos[:, :3].mean(axis=0)
    scale = np.sqrt(3 * np.mean(np.sum(pos[:, :3] ** 2, axis=1)))
    pos[:, :3] /= (scale + 1e-8)

    sensor_type = np.ones(signal.shape[0], dtype=np.int32)

    mean = signal.mean(axis=1, keepdims=True)
    std  = signal.std(axis=1, keepdims=True) + 1e-13
    signal = (signal - mean) / std

    return signal, pos, sensor_type


def to_tensor(seg, p, st, device):
    x       = torch.tensor(seg).unsqueeze(0).to(device)
    pos_t   = torch.tensor(p).unsqueeze(0).to(device)
    stype_t = torch.tensor(st).unsqueeze(0).to(device)
    return x, pos_t, stype_t
