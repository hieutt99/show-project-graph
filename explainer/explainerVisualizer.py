import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.feature_extraction.text import CountVectorizer
import json

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def print(self):
		print(self.x, self.y)

class Textline:
	def __init__(self, points):
		self.points = points
		self.center = Point(0, 0)
		self.xmax = points[0].x
		self.xmin = points[0].x
		self.ymax = points[0].y
		self.ymin = points[0].y
		for i in range(len(points)):
			self.center.x += points[i].x / 4.0
			self.center.y += points[i].y / 4.0
			self.xmax = max(self.xmax, points[i].x)
			self.xmin = min(self.xmin, points[i].x)
			self.ymax = max(self.ymax, points[i].y)
			self.ymin = min(self.ymin, points[i].y)

	def draw_box(self, color='b'):
		n_points = len(self.points)
		for i in range(n_points):
			if i == n_points - 1: draw_line(self.points[i], self.points[0], color)
			else:
				draw_line(self.points[i], self.points[i+1], color)


def draw_line(A, B, color):
	plt.plot([A.x, B.x], [A.y, B.y], color=color)
# def draw_line(img ,A, B, color):
# 	img = cv2.line(img, A, B, color, thickness = 5)


def intersect(x, y, u, v):                      # check if 2 segment [x,y] and [u,v] are intersected to each other
	if x > u: return intersect(u, v, x, y)      # make sure x < u
	return u < y


class imageVisualizer(object):
	def __init__(self, ID, num_labels=5):
		self.ID = ID
		# self.img = mpimg.imread(str('./images/'+ ID + '.jpg'))
		self.img = cv2.imread(str('./images/'+ ID + '.jpg'))
		self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

		self.num_labels = num_labels
		self.labels = ["UNKNOW", "COMPANY", "DATE", "ADDRESS", "TOTAL"]
		self.color = ['k', 'r', 'g', 'b', 'c']  # respectively for unknown, company, date, address, total
		self.textlines = []
		self.edges = [[], [], [], [], []]
		self.features = [[], [], [], [], []]
		self.__get_textline_list()
		self.__get_feature_list()


	def __get_textline_list(self):
		file = open('./part1_output/'+self.ID + '.txt', 'r')
		lines = file.readlines()
		for line in lines:
			splited = line.split(',')
			coors = splited[:8]
			n_points = len(coors) / 2
			points = []
			for k in range(int(n_points)):
				points.append(Point(float(coors[2 * k]), float(coors[2 * k + 1])))
			self.textlines.append(Textline(points))

	def __get_feature_list(self):
		with open('./bow/docs.json', 'r') as fp:
			docs = json.load(fp)
		vectorizer = CountVectorizer(max_features=600)
		vectorizer.fit(docs)
		self.vocab = vectorizer.get_feature_names()

	def add_edge(self, label, edge):
		self.edges[label].append(edge)

	def add_feature(self, label, feature_id):
		self.features[label].append(feature_id)

	def draw_important_box(self, indeces, labels):
		for i in range(len(indeces)):
			self.textlines[indeces[i]].draw_box(color=self.color[labels[i]])

	def draw_edges(self):
		plt.imshow(self.img)
		for label in range(self.num_labels):
			for edge in self.edges[label]:
				draw_line(self.textlines[edge[0]].center, self.textlines[edge[1]].center, color=self.color[label])

	def show(self):
		plt.show()
		for label in range(1, self.num_labels):
			all = ""
			for feature_id in self.features[label]:
				all = all + self.vocab[feature_id] + ", "
			print("Features strongly influenced " + self.labels[label] + ": " + all)
	# def save_image(self):
	# 	cv2.imwrite("./explained_output/"+self.ID+".jpg", self.img)