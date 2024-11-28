from datetime import timedelta
from scipy.io import wavfile
from pyresidfp import SoundInterfaceDevice
from pyresidfp.sound_interface_device import ChipModel
import numpy as np

DELAY_REG = -1
VOICE_REG = -2
VOICES = 3
VOICE_REG_SIZE = 7


def sidq():
    sid = SoundInterfaceDevice()
    return sid.clock_frequency / 1e6 / 1e6


def write_samples(df, name, reg_widths):
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    # max vol
    sid.write_register(24, 15)
    for v in range(3):
        offset = v * VOICE_REG_SIZE
        # max sustain all voices
        sid.write_register(6 + offset, 240)
        # 50% pwm
        sid.write_register(3 + offset, 16)
    raw_samples = []
    df["delay"] = df["diff"] * sidq()
    voice = -1
    for row in df.itertuples():
        if row.reg == VOICE_REG:
            voice = row.val
        elif row.reg != DELAY_REG:
            val = row.val
            reg = row.reg
            width = reg_widths.get(reg, 1)
            if voice >= 0 and reg < VOICE_REG_SIZE:
                reg = (voice * VOICE_REG_SIZE) + reg
            for _ in range(width):
                sid.write_register(reg, val & 255)
                reg += 1
                val >>= 8
        raw_samples.extend(sid.clock(timedelta(seconds=row.delay)))
    wavfile.write(
        name,
        int(sid.sampling_frequency),
        np.array(raw_samples, dtype=np.float32) / 2.0**15,
    )
