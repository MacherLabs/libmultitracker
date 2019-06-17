# MultiTracker
This python library tracks multiple objects in video. It provide simple interface similar to openCV tracker to track multiple objects. It manages multiple single object trackers, dynamic addition of new trackers, identification and removal of obsolete trackers based on detections performed by object detector. It supports use of all OpenCV trackers and dlib correlation tracker for tracking single objects. It is also extensible to easily support custom trackers management like new algorithm for merging new detection with tracking

# Setup
Requirements: dlib, OpenCV, sort, deep_sort
After installing dlib and OpenCV, install multitracker via Pip.

# **How to use**

**Command**:
```python
from multitracker import MultiTracker
import cv2

tracker = MultiTracker(SingleTrackerType = cv2.TrackerKCF_create)	#initialise Multitracker

detections = objectdetector(img)		# get detection from your object detector
obselete_trackers, mapped_trackers, new_detections = tracker.update(img,detections)	# update tracker with new detections.
```
**Input:**
- Detection format is same as OpenCV bbox: **(x,y,w,h)**

**Output:**
- obselete_trackers : list of ids of obsolete single trackers, which have been removed.
- mapped_trackers : dictionary {trackerid: detection index}
- new_detections: list of detections for which new trackers have been created to track them. #TODO: convert format to dictionary {trackerid: detection index}

To use dlib correlation tracker import CorrelationTracker from multitracker. It provide dlib correlation tracker with interface similar to openCV tracker.
```python
from multitracker import MultiTracker,CorrelationTracker

tracker = MultiTracker(SingleTrackerType = CorrelationTracker)	#initialise
```

### Advanced: 
```python
 tracker = MultiTracker( SingleTrackerType, detection2bboxfunc=lambda x:x, removalConfig={})
```
 
 **Specifying tracker deletion criteria:**
 
 It can be specified via **removalConfig** input parameter. It supports removal based on invisible_count, trackers overlap and trackers in corner of image. eg:
 
```python
removalConfig = {
                    'invisible_count':35,       # Remove if no detection found on tracker in 35 countinously tracked frames
                    'overlap_thresh':0.9,       # Remove if no detection found on tracker in overlap_invisible_count countinously tracked frames and trackers bbox is overlaping by 90% (intersect/union)
                    'overlap_invisible_count':15,
                    'corner_percentage':0.1,    # Remove if no detection found on tracker in corner_invisible_count countinously tracked frames and any portion of trackers bbox lies in 10% border oof image 
                    'corner_invisible_count':15
                    }
tracker = MultiTracker( SingleTrackerType, removalConfig=removalConfig)
```
 