from ctypes import cdll

import numpy as np


class YoloX_S:
    def __init__(self):
        self.lib = cdll.LoadLibrary('./yolox_s.so')
        self.out0 = np.zeros((80, 80, 85), dtype=np.float32)
        self.out1 = np.zeros((40, 40, 85), dtype=np.float32)
        self.out2 = np.zeros((20, 20, 85), dtype=np.float32)
        self.lib.init()

    def __del__(self):
        self.lib.destroy()

    def infer(self, image):
        self.lib.infer(
            image.ctypes.data,
            self.out0.ctypes.data,
            self.out1.ctypes.data,
            self.out2.ctypes.data
        )
        return self.out0, self.out1, self.out2

    def infer_thread(self, input: np.ndarray):
        batch_size = input.shape[0]
        out0 = np.zeros((batch_size, 80, 80, 85), dtype=np.float32)
        out1 = np.zeros((batch_size, 40, 40, 85), dtype=np.float32)
        out2 = np.zeros((batch_size, 20, 20, 85), dtype=np.float32)
        self.lib.infer_thread(
            input.ctypes.data,
            out0.ctypes.data,
            out1.ctypes.data,
            out2.ctypes.data,
            batch_size
        )

        return out0, out1, out2
