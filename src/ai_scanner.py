import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.funcs import (bb_intersection_over_union,
                       draw_detected_object,
                       load_settings,
                       draw_stats_dict,
                       ScanLogger)
from src.input_events_simulator import InputEventSimulator


class AIScanner:
    def __init__(self, 
                 settings_file: str = None):

        self.scanned_items: list[dict] = []

        if settings_file is not None:
            settings = load_settings(settings_file)

            # Extraction of parameters
            self.params = {
                "selected_model": settings["selected_model"],
                "region_bbox": settings["region_bbox"],
                "selection_mechanism": settings["frame_selection_mechanism"],
                "denominator": settings["denominator"],
                "range": settings["range"],
                "vizualize_inference": settings["vizualize_inference"],
                "retention_time": settings["retention_time"],
                "vizualization_cooldown_time": settings["vizualization_cooldown_time"],
            }
        else:
            self.params = dict()

    def is_frame_allowed(self, frame_id: int) -> bool:
        """
        Function to check if the frame is allowed to be processed
        Args:
            frame_id (int): The frame number to check
        Returns:
            bool: True if the frame is allowed, False otherwise
        """

        if self.params["selection_mechanism"] == "denominator":
            return frame_id % self.params["denominator"] == 0
        elif self.params["selection_mechanism"] == "range":
            return frame_id in self.params["range"]
        else:
            raise Exception("ERROR: Unknown selection mechanism")

    def extract_roi_objects(self, 
                              reference_bbox: list[int], 
                              zipped_pred: list[tuple[list[int], str, float]],
                              ) -> list[int]:
        """
        Function to extract the objects in the region of interest
        Args:
            reference_bbox (list[int]): The region of interest
            zipped_pred (list[tuple[list[int], str, float]]): The zipped prediction
        Returns:
            list[int]: The list of detected objects
        """

        output = []

        for bbox, class_name, confidence in zipped_pred:
            iou = bb_intersection_over_union(reference_bbox, bbox)
            if iou > 0.1:
                output.append([bbox, class_name, confidence])
        return output
    
    def get_counting_stats(self) -> dict:
        """
        Helper function to get the counting stats from the self.scanned_items
        Returns:
            dict: The counting stats dictionary
        """
        output_dict = dict()
        for this_item in self.scanned_items:
            key = this_item["ai_class"]
            if key not in output_dict:
                output_dict[key] = 1
            else:
                output_dict[key] += 1
        return output_dict

    def count_objects_in_region(self, 
                                video_path: str, 
                                input_event_file: str,
                                output_video_path: str,
                                logging_path: str,
                                ) -> None:
        
        
        # Gaurd clauses for sanity check if the parameters are set
        if not self.params:
            raise Exception("ERROR: No settings provided!")
        
        # Initialize the input event simulator
        input_simulator = InputEventSimulator(input_event_file)

        # Initialize the scan logger
        logger = ScanLogger(logging_path)

        # Update the range of frames to full list from start to end
        self.params["range"] = list(range(self.params["range"][0],
                                          self.params["range"][1]
                                            )
                                    )

        # Function-level variables
        frame_id: int = 0
        last_scaned_frame_id: int = 0

        # Output variable
        self.scanned_items: list[dict] = []

        # Initialize the video capture and video writer
        cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                               cv2.CAP_PROP_FRAME_HEIGHT, 
                                               cv2.CAP_PROP_FPS))
        video_writer: cv2.VideoWriter = cv2.VideoWriter(output_video_path, 
                                                        cv2.VideoWriter_fourcc(*"mp4v"), 
                                                        fps, 
                                                        (w, h))

        #NOTE: TASK-2a Initialize the model
        model = YOLO("model/"+self.params["selected_model"])

        # Variable to hold the last object in ROI through history
        last_roi_object: tuple[str, int] = ("NONE", 0)

        while cap.isOpened():
            logger.frame_id = frame_id

            # start time for the each cycle
            start_time = time.time()

            # Temporary variable to hold the success of reading the frame
            success: bool
            current_image: np.ndarray
            success, current_image = cap.read()

            if not success:
                print("ERROR: Failed to read frame or end of video")
                break
            
            #NOTE: TASK-2b Check if the frame is allowed to be processed and LOGGING
            is_frame_allowed = self.is_frame_allowed(frame_id)
            logger.is_frame_allowed = is_frame_allowed
            if not is_frame_allowed:
                frame_id += 1
                logger.log()
                continue

            print(f"Processing frame_id: {frame_id}")

            # Remove the last_roi_object if the frame is too old
            if last_roi_object[0] != "NONE" and \
                frame_id - last_roi_object[1] > self.params["retention_time"]:
                last_roi_object = ("NONE", 0)

            # Get the simulated item_scan event and LOGGING
            this_event = input_simulator.get_item_scan_event(frame_id)
            logger.is_item_scan_event = this_event["type"] == "item_scan"

            # Visualize the scanning region
            draw_detected_object(frame = current_image,
                                bbox = self.params['region_bbox'],
                                color=(255, 255, 0))
            
            # Perform the inference with tracking
            results = model.track(current_image,
                                  persist=True, 
                                  conf=0.11,
                                  classes = list(range(1,79)), 
                                  tracker='bytetrack.yaml',
                                  verbose=False)

            # Zipping the predictions to get them in pairs of [bbox, class, confidence] and LOGGING
            bboxes = [[int(x) for x in bbox] for bbox in results[0].boxes.xyxy.tolist()]
            classes = [model.names[int(cls)] for cls in results[0].boxes.cls.tolist()]
            confidences = [float(round(conf,2)) for conf in results[0].boxes.conf.tolist()]
            zipped_pred = zip(
                bboxes,
                classes,
                confidences
            )
            
            logger.objects_in_scene = classes

            # Remove the "person and laptop" class from the zipped prediction
            remove_classes = ['book', 'laptop', 'bed', 'suitcase']
            zipped_pred = [x for x in zipped_pred if x[1] not in remove_classes]
            
            #NOTE: TASK-3 Visualize the stats
            stats_dict = self.get_counting_stats()
            draw_stats_dict(current_image,
                            stats_dict = stats_dict,
                            title = "Counting Stats",
                            x = 20,
                            y = 50)

            #NOTE: TASK-4 Draw the detected objects
            for bbox, class_id, confidence in zipped_pred:
                draw_detected_object(current_image, 
                                     bbox, 
                                     class_id+" : "+str(confidence))

            # Get the object which is in ROI and updating the last_roi_object and LOGGING
            roi_objects = self.extract_roi_objects(self.params["region_bbox"], 
                                                        zipped_pred)
            if len(roi_objects) > 0:
                logger.object_in_roi = roi_objects[0][1]
                last_roi_object = [roi_objects[0][1], frame_id]

            # put the last_roi_object on top-right of the current image
            cv2.putText(current_image, 
                        f"Recent ROI item: {last_roi_object[0]}", 
                        (650, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 
                        2)

            # Vizualization of the last item_scan
            if  len(self.scanned_items) > 0 and \
                frame_id - last_scaned_frame_id < self.params["retention_time"]:

                last_object = self.scanned_items[-1]
                draw_stats_dict(current_image,
                                stats_dict = last_object,
                                title = "Last item_scan:",
                                x = 20,
                                y = 250)
        
            # Detecting if the item_scan event is detected and updating the last_scaned_frame_id
            if this_event["type"] == "item_scan":
                last_scaned_frame_id = frame_id

                # push the element in the output list
                this_item = dict({
                    "ai_class": last_roi_object[0],
                    "ai_frame": last_roi_object[1],
                    "scanner_frame": this_event["frame"],
                    "timestamp": this_event["timestamp_str"],
                    "item_id": this_event["item_id"],
                    })
                self.scanned_items.append(this_item)
                
                last_roi_object = ("NONE", 0)

            if self.params["vizualize_inference"]:
                cv2.imshow("counting", current_image)
                cv2.waitKey(1)

            # end time for the whole process and LOGGING
            time_taken_ms = int((time.time() - start_time) * 1000)
            logger.time_taken = time_taken_ms

            video_writer.write(current_image)

            # Increment the frame id
            frame_id += 1
            logger.log()

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        return self.scanned_items