from datetime import timedelta
from scipy.io import wavfile
from pyresidfp import SoundInterfaceDevice
import numpy as np


def sidq():
    sid = SoundInterfaceDevice()
    return sid.clock_frequency / 1e6 / 1e6


def write_samples(df, name):
    sid = SoundInterfaceDevice()
    # max vol
    sid.write_register(24, 15)
    for v in range(3):
        offset = v * 7
        # max sustain all voices
        sid.write_register(4 + offset, 240)
        # 50% pwm
        sid.write_register(3 + offset, 16)
    raw_samples = []
    df["delay"] = df["diff"] * sidq()
    for row in df.itertuples():
        if row.reg != -1:
            val = row.val
            if val > 255:
                sid.write_register(row.reg + 1, val >> 8)
                val = val & 255
            sid.write_register(row.reg, val)
        raw_samples.extend(sid.clock(timedelta(seconds=row.delay)))
    wavfile.write(
        name,
        int(sid.sampling_frequency),
        np.array(raw_samples, dtype=np.float32) / 2.0**15,
    )
