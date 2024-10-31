import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import scipy

from sidtableflip.sidwav import write_samples


class TestSidwav(unittest.TestCase):
    def test_write_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_wav_name = os.path.join(tmpdir, "test.wav")
            test_df = pd.DataFrame(
                [(1, 24, 0), (256, 24, 15)], columns=["diff", "reg", "val"]
            )
            write_samples(test_df, test_wav_name)
            rate, data = scipy.io.wavfile.read(test_wav_name)
            self.assertEqual(rate, 48000)
            data = np.round(data, 2)
            self.assertLess(abs(data.sum()), 0.1)
