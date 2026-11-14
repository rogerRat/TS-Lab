import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = (Path(__file__).resolve().parents[1]
               / "scripts/exp/v1-rage-ads-v2/run_selection.py")
SPEC = importlib.util.spec_from_file_location("v1_rage_ads_v2_selection", MODULE_PATH)
SELECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)


class SimilarityCudaParityTest(unittest.TestCase):
    def test_cuda_matches_cpu_for_threshold_hits_and_scores(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(3, 4096)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        reference = [
            {"vectors": {0: vectors[0], 8: vectors[1]}},
            {"vectors": {3: vectors[2]}},
        ]
        source = [
            {"vectors": {1: vectors[0].copy(), 6: -vectors[1].copy()}},
            {"vectors": {2: vectors[2].copy(), 5: np.zeros(4096, dtype=np.float32)}},
        ]

        cpu = SELECTION._similarity_hits_cpu(reference, source, threshold=0.1)
        cuda = SELECTION._similarity_hits_cuda(reference, source, threshold=0.1)

        self.assertEqual(cuda[0], cpu[0])
        self.assertEqual(cuda[1], cpu[1])
        self.assertEqual(cuda[3], cpu[3])
        for owner in cpu[2]:
            for position, expected in cpu[2][owner].items():
                self.assertAlmostEqual(cuda[2][owner][position], expected, places=5)


if __name__ == "__main__":
    unittest.main()
