import cv2
import torch
import unet
import crnn
import sys
import os
# =================================================================
from torch.utils.data import DataLoader
from graph.dataloader import demo_preprocess, get_page_demo, data_process, ToTensor
from graph.layers import GraphConvolution
from graph.model import GCN
import torch.nn as nn
import torch.optim as optim

import json 
from sklearn.metrics import f1_score, accuracy_score
# =================================================================
from explainer.GNNExplainer import GNNExplainer
from explainer.explainerVisualizer import *
from explainer.converter import *

from graph.model import GCN


import matplotlib.pyplot as plt
# =================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img_folder = './images'
layout_ocr_folder = './part1_output'

list_images = sorted(os.listdir('./images'))

with open('./data/train_list.json','r') as fp:
	train_list = json.load(fp)
print(train_list[1])

dataset = data_process(data_list = train_list[1:2], transform = ToTensor())
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def layout_ocr(img_folder, out_folder='part1_output'):
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)

	unet_model = unet.Model()
	unet_model.load_state_dict(torch.load('weights/unet.pt', map_location='cpu'))

	crnn_model = crnn.Model()
	crnn_model.load_state_dict(torch.load('weights/crnn.pt', map_location='cpu'))

	img_names = os.listdir(img_folder)

	for img_name in img_names:
		im = os.path.join(img_folder, img_name)
		im = cv2.imread(im)

		m, h = unet.segment(unet_model, im)
		boxes = unet.detect_lines(m, h, im.shape)

		label_file = open(os.path.join(out_folder, img_name[:-3] + 'txt'), 'w')

		for box in boxes:
			x1, y1 = box[0]
			x2, y2 = box[1]

			text_line = im[y1:y2, x1:x2]
			text_line = cv2.cvtColor(text_line, cv2.COLOR_BGR2GRAY)
			text = crnn.ocr(crnn_model, text_line)

			if text == '':
				continue

			for c in (x1, y1, x2, y1, x2, y2, x1, y2):
				label_file.write(str(c) + ', ')
			label_file.write(text)
			label_file.write('\n')

		label_file.close()

# ===============================================================================
# print("Layout - OCR ... ")
# # layout_ocr(img_folder, layout_ocr_folder)
# print("Organizing ... ")
# list_files = sorted(os.listdir(layout_ocr_folder))
# print(list_files)

# graph_dataset = demo_preprocess(data_list = list_files[1:2])

# for i, batch in enumerate(graph_dataset):
# 	print(batch[0].size(), batch[1].size())


# todo: Load trained GCN
print("Loading trained GCN ...")
device = 'cpu'
model = GCN()
model.load_state_dict(torch.load('./model12.pth.tar', map_location=torch.device('cpu')))
print(model)
print('Done!')

# ==============================================================================================

def use_gcn(model, dataset, device):
	model.to(device)

	for item in dataset:
		[V, A] = item
		V = V.to(device)
		A = A.to(device)

		with torch.no_grad():
			output = model.eval([V, A])
			return output

def process_image(image, pos, label):
	# Box the bboxes
	for i, item in enumerate(pos):
		if label[i] == 0:
			color = (0,0,0) # black
		if label[i] == 1:
			color = (255,0,0) # red - company
		if label[i] == 2:
			color = (0,255,0) # green - date
		if label[i] == 3:
			color = (0,0,255) # blue - address
		if label[i] == 4:
			color = (0,255,255) # cyan - total
		image = cv2.rectangle(image, (item[0],item[1]), (item[4],item[5]), color = color, thickness = 2)

	return image

# out = use_gcn(model, graph_dataset, device)
# prediction = out[0].numpy().argmax(axis = 1)

# print(prediction)

# image = cv2.imread('./images/'+list_images[3])
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# s, pos = get_page_demo(list_files[3])
# for i in s:
# 	print(i)

# image = process_image(image, pos, prediction)
# plt.imshow(image)
# plt.show()

# ======================================================================================

# todo: Demo
WANT_TO_TEST = [0]
TEST_ID = 'X00016469612'
explainer = GNNExplainer(model, epochs=200, lr=0.01)

# ==============================================
# list_id = []
# for item in list_files:
# 	list_id.append(item[:-4])
# list_id = sorted(list_id)
# ==============================================

for i_test, (sample, name) in enumerate(zip(dataset, train_list[1:2])):
	# if i_test not in WANT_TO_TEST: continue

	# todo: Load pre-processed data
	"""
		V (Tensor): N x F  --> number of nodes x number of features 	(this case: N x 600)
		A (Tensor): L x N x N (this case: L = 4)
		edge_index (Tensor): 2 x number_of_edges
		edge_type (Tensor): number_of_edges
	"""
	[V, A], label = sample
	edge_index, edge_type = convert_adj_to_edge_index(A)
	# print(edge_index)

	# todo: calculate predicted labels of nodes --> identify which nodes to be explained
	with torch.no_grad():
		pred = model.eval([V.to(device), A.to(device)])
	pred = pred[0].argmax(axis=1).cpu().numpy()
	print(pred)

	indeces_of_nodes_to_explained = []
	values_of_nodes_to_explained = []
	for i in range(pred.shape[0]):
		if pred[i] != 0:
			indeces_of_nodes_to_explained.append(i)
			values_of_nodes_to_explained.append(pred[i])

	print("Nodes to be explained: ", indeces_of_nodes_to_explained)
	print("With respective label: ", values_of_nodes_to_explained)

	# todo: explain each nodes
	vis = imageVisualizer_temp(name)        # change the ID of the test here!!!
	edge_threshold = 0.9

	for node_id in indeces_of_nodes_to_explained:
		feature_mask, edge_mask = explainer.explain_node(node_id, V, A)
		"""
			feature_mask (tensor): size = number_of_features; value in range[0,1]
			edge_mask (tensor): size = number_of_edge; value in range[0,1]
		"""
		for z in range(edge_mask.size(0)):
			if edge_mask[z] > edge_threshold:
				# print(pred[node_id], int(edge_index[0][z]), int(edge_index[1][z]))
				vis.add_edge(pred[node_id], (int(edge_index[0][z]), int(edge_index[1][z])))

		print(feature_mask.min(), feature_mask.max())
		important_features = take_n_most_important(feature_mask)
		for feature_id in important_features: vis.add_feature(pred[node_id], feature_id)

	vis.draw_important_box(indeces_of_nodes_to_explained, values_of_nodes_to_explained)
	vis.draw_edges()
	vis.show()
	# vis.save_image()

print("Explained")
