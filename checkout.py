import json
from src.ai_scanner import AIScanner

if __name__ == "__main__":
    scanner = AIScanner("settings.yaml")

    # Forcefully set the parameters for unit testing
    # scanner = AIScanner()
    # scanner.params = dict(
    #     {
    #         "selected_model": "yolo11s.pt",
    #         "region_bbox": [370, 30, 630, 200],
    #         "selection_mechanism": "denominator",
    #         "denominator": 1,
    #         "range": [40, 180],
    #         "vizualize_inference": True,
    #         "retention_time": 30,
    #         "vizualization_cooldown_time": 60,
    #     }
    # )

    scanned_items = scanner.count_objects_in_region("data/cam_2.mp4",
                                                    "data/input_events.txt",
                                                    "output/output_cam_2.mp4",
                                                    "output/output_logging.csv")
    
    for item in scanned_items:
        print(item)

    #NOTE: TASK-3 (optional) write the scanned_items to a file for later usage
    with open("output/output_scanned_items.json", "w") as f:
        json.dump(scanned_items, f)
    
