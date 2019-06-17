import dlib

def cvbox2drectangle(bbox):
	return dlib.drectangle(*(bbox[0],bbox[1],bbox[2]+bbox[0],bbox[3]+bbox[1]))

def drectangle2cvbox(rect):
	p1 =(int(rect.left()), int(rect.top()))
	return p1+(int(rect.right())-p1[0], int(rect.bottom())-p1[1])

def union(a,b):
	x = min(a[0], b[0])
	y = min(a[1], b[1])
	w = max(a[0]+a[2], b[0]+b[2]) - x
	h = max(a[1]+a[3], b[1]+b[3]) - y
	return (x, y, w, h)

def area(box):
	return box[2]*box[3]