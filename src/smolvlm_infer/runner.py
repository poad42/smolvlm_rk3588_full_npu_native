import os
import numpy as np
from rknnlite.api import RKNNLite


class RKNNBlockRunner:
    def __init__(self, path, core_id=None):
        self.name = os.path.basename(path)
        self.rknn = RKNNLite(verbose=False)
        self.rknn.load_rknn(path)
        mask = RKNNLite.NPU_CORE_AUTO
        if core_id == 0:
            mask = RKNNLite.NPU_CORE_0
        if core_id == 1:
            mask = RKNNLite.NPU_CORE_1
        if core_id == 2:
            mask = RKNNLite.NPU_CORE_2
        self.rknn.init_runtime(core_mask=mask)

    def run(self, x):
        # Strict 1024 padding/cropping logic
        if x.shape[1] < 1024:
            pad = np.zeros((1, 1024 - x.shape[1], 768), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        elif x.shape[1] > 1024:
            x = x[:, :1024, :]
        return self.rknn.inference(inputs=[x])[0]
