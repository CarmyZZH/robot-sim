import unittest
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from models.model import MorphGNN, MorphGNNLite
from utils.graph_converter import RobotGraphConverter


class MorphGraphContractTests(unittest.TestCase):
    def test_morph_graph_matches_feature_node_count(self):
        converter = RobotGraphConverter()
        features = converter.get_feature_matrix(1.5)
        edge_index = converter.edge_index_7

        self.assertEqual(features.shape, (7, 10))
        self.assertEqual(tuple(edge_index.shape), (2, 12))
        self.assertEqual(int(edge_index.min()), 0)
        self.assertEqual(int(edge_index.max()), features.shape[0] - 1)

    def test_graph_to_environment_joint_order(self):
        graph_angles = np.arange(7, dtype=np.float32)

        env_angles = RobotGraphConverter.graph_to_env_angles(graph_angles)

        self.assertEqual(env_angles, [2, 1, 0, 3, 4, 5, 6])

    def test_full_and_lite_models_share_inference_contract(self):
        converter = RobotGraphConverter()
        x = torch.tensor(converter.get_feature_matrix(1.5))
        edge_index = converter.edge_index_7

        for model in (MorphGNN(), MorphGNNLite()):
            final_output, raw_output = model(x, edge_index)
            predicted = model.predict(x, edge_index)

            self.assertEqual(tuple(final_output.shape), (7, 1))
            self.assertEqual(tuple(raw_output.shape), (7, 1))
            self.assertEqual(tuple(predicted.shape), (7, 1))


if __name__ == "__main__":
    unittest.main()
