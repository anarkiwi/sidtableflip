from datetime import timedelta
from scipy.io import wavfile
from pyresidfp import SoundInterfaceDevice
import numpy as np


def write_samples(df, name, diffpad=8):
    sid = SoundInterfaceDevice()
    df["secs"] = (df["diff"] + diffpad) * (sid.clock_frequency / 1e6) / 1e6
    sid.write_register(24, 15)
    raw_samples = []
    for row in df.itertuples():
        raw_samples.extend(sid.clock(timedelta(seconds=row.secs)))
        sid.write_register(row.reg, row.val)
    wavfile.write(
        name,
        int(sid.sampling_frequency),
        np.array(raw_samples, dtype=np.float32) / 2.0**15,
    )
