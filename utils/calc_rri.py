
import numpy as np
from ecgdetectors import Detectors



def get_rri(sig, fs=300, n_beats=10):
    """
    Input: 10s ECG signal (torch tensor)
    Output: RRI, time in between each beat (10 beats otherwise 0-padded)
    """
    detectors = Detectors(fs)
    sig_n = sig.numpy()
    # if type(sig).__module__ != np.__name__:
    #  sig_n = sig.numpy()
    # else:
    # sig_n = sig
    # print(sig_n.shape)
    rri_list = []
    for i in range(sig_n.shape[0]):
        r_peaks = detectors.pan_tompkins_detector(sig_n[i])
        rri = np.true_divide(np.diff(r_peaks), fs)
        if len(rri) < n_beats:
            rri = np.pad(rri, (0, n_beats - len(rri)), 'constant', constant_values=(0))
        if len(rri) > n_beats:
            rri = rri[0:n_beats]
        rri_list.append(rri)

    rri_stack = np.stack(rri_list, axis=0)
    # print(rri_stack.shape)
    return rri_stack


if __name__ == '__main__':
    pass