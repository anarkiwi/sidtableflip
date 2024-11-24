from datetime import timedelta
from scipy.io import wavfile
from pyresidfp import SoundInterfaceDevice
from pyresidfp.sound_interface_device import ChipModel
import numpy as np

DELAY_REG = -1
REG_BITSHIFT = {
    0: 8,
    2: 4,
    7: 8,
    9: 4,
    14: 8,
    16: 4,
    21: 3,
}


def sidq():
    sid = SoundInterfaceDevice()
    return sid.clock_frequency / 1e6 / 1e6


def write_samples(df, name):
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    # max vol
    sid.write_register(24, 15)
    for v in range(3):
        offset = v * 7
        # max sustain all voices
        sid.write_register(6 + offset, 240)
        # 50% pwm
        sid.write_register(3 + offset, 16)
    raw_samples = []
    df["delay"] = df["diff"] * sidq()
    for row in df.itertuples():
        if row.reg != DELAY_REG:
            val = row.val
            bits = REG_BITSHIFT.get(row.reg, None)
            if bits is None:
                sid.write_register(row.reg, val)
            else:
                lo = val & 2**bits - 1
                hi = val >> bits
                sid.write_register(row.reg, lo)
                sid.write_register(row.reg + 1, hi)
        raw_samples.extend(sid.clock(timedelta(seconds=row.delay)))
    wavfile.write(
        name,
        int(sid.sampling_frequency),
        np.array(raw_samples, dtype=np.float32) / 2.0**15,
    )
