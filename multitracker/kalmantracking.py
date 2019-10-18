import numpy as np
import os
# from numba import jit

WORK_DIR = "/LFS/libmultitracker"
MODEL_DIR = 'models'

INVISIBLE_THRESH = 30                               #param

def to_cvbox(detections, classes):
    return [
        (int(det['box']['topleft']['x']), int(det['box']['topleft']['y']), 
        int(det['box']['bottomright']['x']), int(det['box']['bottomright']['y']),
        det["prob"])
        for det in detections if det['class'] in classes
    ]

class Tracker(object):
    def __init__(self):
        self.bbox = []
        self.consecutive_invisible_count = 0

class KalmanTracker(object):
    def __init__(self, classes, tracker='sort', gpu_config = 0.02):
        self.ttype = tracker
        self.classes = classes
        if tracker == 'deep_sort':
            from deep_sort import generate_detections
            from deep_sort.deep_sort import nn_matching
            from deep_sort.deep_sort.tracker import Tracker
            
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 100)                      #param
            self.nms_max_overlap = 0.1                                                                  #param
            model_path = os.path.join(WORK_DIR, MODEL_DIR, "mars-small128.ckpt-68577")
            self.encoder = generate_detections.create_box_encoder(model_path, gpu_config = gpu_config)
            self.tracker = Tracker(metric)

            from deep_sort.application_util import preprocessing as prep
            from deep_sort.deep_sort.detection import Detection
            self.prep = prep
            self.Detection = Detection

        elif tracker == 'sort':
            from sort.sort import Sort
            self.tracker = Sort()

        self.trackers = {}

    def update(self, imgcv, detections):
        boxes = to_cvbox(detections, self.classes)
        detections, scores = [], []
        ids, bboxes = [], []

        for b in boxes:     
            left, top, right, bot, confidence = b
            if self.ttype == 'deep_sort':
                detections.append(np.array([left,top,right-left,bot-top]).astype(np.float64))
                scores.append(confidence)
            elif self.ttype == 'sort':
                detections.append(np.array([left,top,right,bot]).astype(np.float64))
        
        if self.ttype == "deep_sort":
            self.tracker.predict()
        
        detections = np.array(detections)
        if detections.shape[0] == 0 :
            self.check_obsolete()
            return

        if self.ttype == "deep_sort":
            scores = np.array(scores)
            features = self.encoder(imgcv, detections.copy())
            detections = [
                        self.Detection(bbox, score, feature) for bbox,score, feature in
                        zip(detections,scores, features)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = self.prep.non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            self.tracker.update(detections)
            trackers = self.tracker.tracks

        elif self.ttype == "sort":
            trackers = self.tracker.update(detections)

        for track in trackers:
            if self.ttype == "deep_sort":
                if not track.is_confirmed() or track.time_since_update > 1:             #param
                    continue
                bbox = track.to_tlbr()
                bbox = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
                id_num = int(track.track_id)
                self.add_trackers(id_num, bbox)
            elif self.ttype == "sort":
                bbox = [track[0],track[1],track[2]-track[0],track[3]-track[1]]
                id_num = int(track[4])
                self.add_trackers(id_num, bbox)
        
        self.check_obsolete()
        # print len(self.trackers)

    def add_trackers(self, id_num, bbox):
        tracker = self.trackers.get(id_num, Tracker())
        tracker.bbox = bbox
        tracker.consecutive_invisible_count = 0
        self.trackers[id_num] = tracker

    # @jit
    def check_obsolete(self):
        for id_num, tracker in self.trackers.items():
            tracker.consecutive_invisible_count += 1
            if tracker.consecutive_invisible_count > INVISIBLE_THRESH:
                del self.trackers[id_num]
