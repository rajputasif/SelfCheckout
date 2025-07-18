import unittest
from src.ai_scanner import AIScanner

class TestStringMethods(unittest.TestCase):

    def test_checkout(self):
        scanner = AIScanner()
        scanner.params = dict(
            {
                "selected_model": "yolo11s.pt",
                "region_bbox": [370, 30, 630, 200],
                "selection_mechanism": "range",
                "denominator": 1,
                "range": [40, 200],
                "vizualize_inference": False,
                "retention_time": 30,
                "vizualization_cooldown_time": 60,
            }
        )

        scanned_items = scanner.count_objects_in_region("data/cam_2.mp4",
                                                        "data/input_events.txt",
                                                        "output/output_cam_2.mp4",
                                                        "output/output_logging.csv")
        print(scanned_items)
        
        ground_truth_labels = ["bottle", "cell phone", "NONE"]
        for i,gt in enumerate(ground_truth_labels):
            self.assertIn(gt, scanned_items[i]["ai_class"])

    def test_latency(self):
        allowed_latency = 30

        scanner = AIScanner()
        scanner.params = dict(
            {
                "selected_model": "yolo11s.pt",
                "region_bbox": [370, 30, 630, 200],
                "selection_mechanism": "range",
                "denominator": 1,
                "range": [40, 200],
                "vizualize_inference": False,
                "retention_time": 30,
                "vizualization_cooldown_time": 60,
            }
        )

        scanned_items = scanner.count_objects_in_region("data/cam_2.mp4",
                                                        "data/input_events.txt",
                                                        "output/output_cam_2.mp4",
                                                        "output/output_logging.csv")
        print(scanned_items)
        
        for item in enumerate(scanned_items):
            if item['ai_class'] == "NONE":
                continue
            self.assertGreater(allowed_latency, 
                            item['scanner_frame']-item['ai_frame'], 
                            "Latency is greater than allowed latency")
        

if __name__ == '__main__':
    unittest.main()