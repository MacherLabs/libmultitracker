from singletracker import getSingleTracker
import numpy as np
from utils import *
import logging
import cv2
import itertools
import random
logger = logging.getLogger("Multitracker")

class MultiTracker:
	def __init__(self, SingleTrackerType, detection2bboxfunc=lambda x:x, removalConfig={}):
		self.trackers={}
		self.detection2bbox = detection2bboxfunc
		self.SingleTracker = getSingleTracker(SingleTrackerType, self.detection2bbox)
		self.removalConfig = {'invisible_count':35,'overlap_thresh':0,'overlap_invisible_count':0,'corner_percentage':0,'corner_invisible_count':0}
		self.removalConfig.update(removalConfig)
		
	def update(self, img, detections = []):
		# update_alltrackers()
		for trackerid in self.trackers.keys():
			# timer = cv2.getTickCount()
			(self.trackers[trackerid].confidence, self.trackers[trackerid].bbox) = self.trackers[trackerid].tracker.update(img)
			# fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);logger.info("fps: {}".format(fps))
			self.trackers[trackerid].age +=1
			self.trackers[trackerid].consecutive_invisible_count +=1
		
		logger.debug("Processing detections---")
		# obselete,old,mapped,new = merge_trackers(self,img,detections)		
		obselete_trackers, mapped_trackers, new_detections= self.merge_trackers(img,detections)

		# update_merged(mapped)
		self.update_trackers_with_detections(img, detections, mapped_trackers)
		# add_trackers(new)
		# self.add_trackers(img,new_detections)
		# remove_trackers(obselete)
		self.remove_trackers(obselete_trackers)

		return obselete_trackers, mapped_trackers, new_detections 
	
	def add_trackers(self, img, detections):
		# Provide the tracker the initial position of the object
		for i, detection in enumerate(detections):
			trackerID =self.new_trackerID(detection)
			self.trackers[trackerID] = self.SingleTracker(trackerID, img, detection) #TODO , detection2bbox=self.detection2bbox)
	
	def add_tracker(self, img, detection):
		# Provide the tracker the initial position of the object
		trackerID =self.new_trackerID(detection)
		self.trackers[trackerID] = self.SingleTracker(trackerID, img, detection)
		return trackerID
		
	def update_trackers_with_detections(self,img, detections, mapped_trackers):
		for trackerid,detections in mapped_trackers.items():
			detection = detections['box']
			self.trackers[trackerid].tracker.init(img, self.detection2bbox(detection)) 
			self.trackers[trackerid].visible_count +=1
			self.trackers[trackerid].consecutive_invisible_count = 0


	def remove_trackers(self,obselete_trackers):
		for trackerid in obselete_trackers:
			self.trackers.pop(trackerid,None)

	def merge_trackers(self, img, detections):
		obselete_trackers	= []
		mapped_trackers	= {}
		new_detections 	= {}

		points = map(self.detection2bbox,detections)
		trackers = self.trackers

		### Main Algo here
		if detections:
			##Calculate cost and minimize it:
			Cost = np.zeros((len(points),len(trackers)))
			for i,p in enumerate(points):
				prect = cvbox2drectangle(p)
				for j,trackerid in enumerate(trackers):
					Cost[i,j] = prect.intersect(cvbox2drectangle(trackers[trackerid].bbox)).area()/area(union(p,trackers[trackerid].bbox))
			Minimun_cost_index = np.argmax(Cost, axis=0)

			## detections mapped to trackers:
			covered_points = []
			for j,trackerid in enumerate(trackers):
				i = Minimun_cost_index[j]
				if i  not in covered_points:
					if Cost[i,j] > 0.5:
						covered_points.append(i)
						mapped_trackers[trackerid] = {"index":i,"box":detections[i]}

			##Add new tracker for new detections:	
			for i in range(len(points)):
				if i not in covered_points:
					trackerid=self.add_tracker(img,detections[i])
					new_detections[trackerid] = {"index":i,"box":detections[i]}		
					mapped_trackers[trackerid] = {"index":i,"box":detections[i]}					

		##Obselete_trackers:
		# tracker removal by invisble_count
		if self.removalConfig['invisible_count']:
			invisible_count_thresh = self.removalConfig['invisible_count']
			removal_candidate_trackers = set(trackers.keys())- set(mapped_trackers.keys()) - set(obselete_trackers)
			for i in removal_candidate_trackers:
				if not trackers[i].confidence or trackers[i].consecutive_invisible_count >invisible_count_thresh:
					obselete_trackers.append(i)
					logger.debug("tracker removal by invisble_count:{}".format(i))

		# tracker removal by overlapping threshold:
		if  self.removalConfig['overlap_thresh']:
			removal_thresh = self.removalConfig['overlap_thresh']
			overlap_invisible_count = self.removalConfig['overlap_invisible_count']
			removal_candidate_trackers = list(removal_candidate_trackers - set(obselete_trackers))
			valid_trackers =  removal_candidate_trackers+mapped_trackers.keys()

			if valid_trackers:
				valid_trackers_bbox = np.array([ trackers[idx].bbox for idx in valid_trackers]) 	# x,y,w,h
				valid_trackers_bbox[:,2:] += valid_trackers_bbox[:,:2]							# x1,y1,x2,y2 

				for idx,trackerid in enumerate(removal_candidate_trackers):
					idxplus = idx+1
					np_bbox = np.array(trackers[trackerid].bbox)
					intersect_x1 = np.maximum(np_bbox[0],valid_trackers_bbox[idxplus:,0])
					intersect_y1 = np.maximum(np_bbox[1],valid_trackers_bbox[idxplus:,1])
					intersect_x2 = np.minimum(np_bbox[2],valid_trackers_bbox[idxplus:,2])
					intersect_y2 = np.minimum(np_bbox[3],valid_trackers_bbox[idxplus:,3])
					intersect_area = (intersect_x2-intersect_x1) * (intersect_y2 - intersect_y1)

					union_x1 = np.minimum(np_bbox[0],valid_trackers_bbox[idxplus:,0])
					union_y1 = np.minimum(np_bbox[1],valid_trackers_bbox[idxplus:,1])
					union_x2 = np.maximum(np_bbox[2],valid_trackers_bbox[idxplus:,2])
					union_y2 = np.maximum(np_bbox[3],valid_trackers_bbox[idxplus:,3])
					union_area = (union_x2-union_x1) * (union_y2 - union_y1)

					overlap_check = (intersect_area/union_area >removal_thresh)

					if overlap_check.any():
						if trackers[trackerid].consecutive_invisible_count > overlap_invisible_count:
							obselete_trackers.append(trackerid)
							logger.debug("tracker removal by overlap_thresh:{}".format(trackerid))

		# corner tracker removal by invisible_count threshold:
		if self.removalConfig['corner_invisible_count']:
			if not self.removalConfig['overlap_thresh']:
				removal_candidate_trackers = list(removal_candidate_trackers - set(obselete_trackers))
				valid_trackers =  removal_candidate_trackers+mapped_trackers.keys()

				if valid_trackers:
					valid_trackers_bbox = np.array([ trackers[idx].bbox for idx in valid_trackers]) 	# x,y,w,h
					valid_trackers_bbox[:,2:] += valid_trackers_bbox[:,:2]							# x1,y1,x2,y2 

			if valid_trackers:
				corner_invisible_count = self.removalConfig['corner_invisible_count']
				img_shape  = img.shape[:2]
				corner_x1,corner_y1 = int(img_shape[1]*self.removalConfig['corner_percentage']),int(img_shape[0]*self.removalConfig['corner_percentage'])  # 10% 
				corner_x2,corner_y2 = (img_shape[1]- corner_x1, img_shape[0]-corner_y1)
				removal_candidate_trackers_bbox = valid_trackers_bbox[:len(removal_candidate_trackers)]
				corner_check = (removal_candidate_trackers_bbox[:,0] < corner_x1 ) | (removal_candidate_trackers_bbox[:,1] < corner_x1 ) | \
									(removal_candidate_trackers_bbox[:,2] > corner_x2 ) | (removal_candidate_trackers_bbox[:,3] > corner_y2 )

				print corner_x1,corner_y1,corner_x2,corner_y2

				# import  pdb;pdb.set_trace()
				for trackerid in itertools.compress(removal_candidate_trackers,corner_check):
					if trackers[trackerid].consecutive_invisible_count > corner_invisible_count:
						obselete_trackers.append(trackerid)
						logger.debug("tracker removal by corner_percentage:{}".format(trackerid))

		#
		obselete_trackers = list(set(obselete_trackers))


		## Main algo finish
		return obselete_trackers, mapped_trackers, new_detections

	def new_trackerID(self, detection):
		return random.randint(0,1000)
		# for i in range(100):
		# 	if i not in self.trackers:
		# 		return i
