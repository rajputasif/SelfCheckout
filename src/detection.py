import cv2
from ultralytics import YOLO

# Function to print detected object of [x, y, w, h, class] on image
def print_detected_object(frame, detected_object):
    x, y, x2, y2, detected_class = detected_object

    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{detected_class}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)   

def count_objects_in_region(video_path, output_video_path, model_path):
    last_detected = "NONE"
    count_dict = dict()

    frame_id = 0

    """Count objects in a specific region within a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_bbox = [(350, 200), (630, 200), (630, 20), (350, 20)]

    model = YOLO("yolo11s.pt")

    # print the classes
    print(model.names)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        
        results = model(im0)
        annotated_frame = results[0].plot()

        # print the detected objects except for the class "person"
        for i, detected_class in enumerate(results[0].boxes.cls.tolist()):
            this_object_class = model.names[int(detected_class)]
            bbox = [int(x) for x in results[0].boxes.xyxy[i].tolist()]
            if this_object_class != "person":
                bbox_obj = [bbox[0], bbox[1], bbox[2], bbox[3], this_object_class]

                print("Detected object: ", this_object_class, bbox_obj)

                print_detected_object(im0, bbox_obj)

        print(f"Frame {frame_id} has been processed")

        if frame_id == 80 or frame_id == 125 or frame_id == 180:
            print("item_scan detected")
            cv2.waitKey(0)

        frame_id += 1
        
        cv2.imshow("counting",im0)
        cv2.waitKey(1)

        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("cam_2.mp4", "output_cam_2.mp4", "yolo11s.pt")