import unittest
from pathlib import Path
import config

class ProjectTests(unittest.TestCase):
    def test_model_url_exists(self):
        self.assertTrue(config.MODEL_URL.startswith("https://"))

    def test_model_path_parent(self):
        self.assertEqual(Path(config.MODEL_PATH).parent.name, "models")

    def test_threshold(self):
        self.assertGreaterEqual(config.DEFAULT_THRESHOLD, 0)
        self.assertLessEqual(config.DEFAULT_THRESHOLD, 1)

if __name__ == "__main__":
    unittest.main()
