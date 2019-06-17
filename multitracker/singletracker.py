import dlib
import cv2
from utils import *
# tracker = cv2.TrackerKCF_create()

# tracker = dlib.correlation_tracker()
def getSingleTracker(TrackerType, detection2bbox=lambda x:x):
	class SingleTracker(object):
		def __init__(self,Id, img, detection):
			self.id = Id
			self.age = 1
			self.visible_count = 1
			self.consecutive_invisible_count = 0
			self.detection2bbox = detection2bbox	# TODO
			self._last_detection = detection
			self.bbox = detection2bbox(detection)	 # Can be converted for numpy format
			self.confidence = 0
			self.tracker = TrackerType()
			self.tracker.init(img, self.bbox)		
	
		def update(self,img,detection=None):
			if detection:
				retVal = self.tracker.update(img,detection2bbox(detection))
			else:
				retVal = self.tracker.update(img)
			return retVal

		def isObselete(self):
			pass

	return SingleTracker

class CorrelationTracker(dlib.correlation_tracker):
	"""docstring for correlation_tracker"""
	def __init__(self):
		super(CorrelationTracker, self).__init__()

	def init(self,img, rect):
		self.start_track(img, cvbox2drectangle(rect))

	def update(self,img,bbox=None):	# -> confidence, bbox , returns similar to openCV tracker
		if bbox:
			return (super(CorrelationTracker, self).update(img,cvbox2drectangle(bbox)), drectangle2cvbox(self.get_position()))
		else:
			return (super(CorrelationTracker, self).update(img), drectangle2cvbox(self.get_position()))
