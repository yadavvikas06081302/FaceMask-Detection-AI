import sys
import types
import unittest

# Keep the label-safety tests lightweight; Ultralytics itself is tested by the
# integration/runtime checks on a machine where the ML dependencies are installed.
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

from mask_detector import classify_mask_label, normalize_class_name


class TestMaskLabels(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_class_name("Mask-Weared Incorrect"), "mask_weared_incorrect")

    def test_common_mask_labels(self):
        self.assertEqual(classify_mask_label("with_mask"), "Mask")
        self.assertEqual(classify_mask_label("mask"), "Mask")
        self.assertEqual(classify_mask_label("without_mask"), "No Mask")
        self.assertEqual(classify_mask_label("no-mask"), "No Mask")
        self.assertEqual(classify_mask_label("mask_weared_incorrect"), "Mask Incorrect")

    def test_unknown_is_not_guessed(self):
        self.assertEqual(classify_mask_label("person"), "Unknown")


if __name__ == "__main__":
    unittest.main()
