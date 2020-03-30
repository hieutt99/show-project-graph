import torch
from torch.utils.data import DataLoader
from graph.dataloader import *
from graph.layers import GraphConvolution
from graph.model import GCN
import torch.nn as nn
import torch.optim as optim

import json 
import cv2
from sklearn.metrics import f1_score, accuracy_score

print("========================================")
print("~~~~~~~~~~~~~~GCN~~~~~~~~~~~~~~~~~~~~~~~")
with open('./data/train_list.json','r') as fp:
	train_list = json.load(fp)


dataset = data_process(data_list = train_list[:], transform = ToTensor())
# dataloader = DataLoader(dataset, shuffle = True)
# for i, sample in enumerate(dataset):
# 	print(sample[0][0].shape, sample[0][1].shape, sample[1].shape)
# 	[V, A], label = sample
# 	temp = V.numpy()
# 	for i in temp:
# 		for  j in i:
# 			if j != 0:
# 				print(j, end = '')
# 		print()
# g = GraphConvolution(3896, 3896*2, 2, 50)

model = GCN()
device = torch.device('cpu')
# device = torch.device('cuda:0')

def train(model, dataset, device = 'cpu', save_path = None, n_epochs = 10, lr = 0.1, momentum = 0.9):


	model.to(device)
	optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
	# optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=0.9)

	for epoch in range(n_epochs):
		running_loss = 0.0
		for i, data in enumerate(dataset):
			inputs, labels = data
			# print(inputs[0].size())
			# inputs = [inputs[0].to(device), inputs[1].to(device)]
			labels = labels.to(device)
			inputs = [inputs[0].to(device), inputs[1].to(device)]
			
			optimizer.zero_grad()

			outputs = model(inputs)

			loss = model.criterion(outputs[0], labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

		print(f"Epoch {epoch+1}/{n_epochs} - Loss : {running_loss}-{running_loss/dataset.__len__()}")
		# print(f"Epoch {epoch+1} - Loss : {running_loss}")
	print("Finished running")

	if save_path:
		torch.save(model.state_dict(), save_path)
		print('Saved model')
	
	return model

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test(model, dataset, device):
	'''
	total line acc = sum(each line 0 or 1)/total
	'''
	count = 0
	total_pages = dataset.__len__()
	total_lines = 0

	macro = 0
	micro = 0
	acc = 0

	model.to(device)
	for [V,A], label in dataset:
		V = V.to(device)
		A = A.to(device)
		label = label.to(device)
		total_lines += V.size(0)

		with torch.no_grad():
			outputs = model.eval([V, A])
			predict = outputs[0].argmax(axis = 1)

			count += (predict == label).cpu().sum().item()

			macro += f1_score(predict.cpu().numpy(), label.cpu().numpy(), average = 'macro')
			micro += f1_score(predict.cpu().numpy(), label.cpu().numpy(), average = 'micro')
			acc += accuracy_score(predict.cpu().numpy(), label.cpu().numpy())

	print('Accuracy %f' %(count/total_lines))
	print('Accuracy %f' %(acc/total_pages))
	print('Macro F1 : %f - %f' %(macro/total_pages, macro))
	print('Micro F1 : %f - %f' %(micro/total_pages, micro))
	print(count)
	print(total_lines)

def significant_test(model, dataset, device):
	'''
	total line acc = sum(each line 0 or 1)/total
	'''

	total_significant = 0
	significant_point = 0
	s_count = 0

	model.to(device)
	for [V,A], label in dataset:
		V = V.to(device)
		A = A.to(device)
		label = label.to(device)
		# total_significant += torch.sum(label.)

		with torch.no_grad():
			outputs = model.eval([V, A])
			predict = outputs[0].argmax(axis = 1)

			temp = (predict == label).cpu().numpy()
			x = label.nonzero()
			# print(x)
			# print(temp)
			total_significant += x.size(0)
			for i in x:
				if temp[i] == True:
					significant_point += 1

	print('Accuracy significant point %f' %(significant_point/total_significant))
	print(significant_point)
	print(total_significant)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(model)
# model.load_state_dict(torch.load('model12.pth.tar'))
model.load_state_dict(torch.load('model12.pth.tar', map_location=device))

test(model, dataset, device)
significant_test(model, dataset, device)
# model = train(model, dataset, device = device, save_path = 'model12.pth.tar', lr = 0.001, n_epochs = 100)
# test(model, dataset, device)
# significant_test(model, dataset, device)
# model = train(model, dataset, device = device, save_path = 'model12.pth.tar', lr = 0.001, n_epochs = 100)
# test(model, dataset, device)
# significant_test(model, dataset, device)


for i, sample in enumerate(dataset):
	[V,A], label = sample
	if i == 2:
		break
with torch.no_grad():
	pred = model.eval([V.to(device),A.to(device)])
pred = (pred[0]).argmax(axis = 1).cpu().numpy()
print(pred)
print(label.cpu().numpy())


# ========================================================


def process_image(image, pos, center, label):
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


tmp_link = "./data/task2train_pro/"+train_list[2]+".jpg"
image = cv2.imread(tmp_link)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

s, pos, label = get_page(train_list[2])

centers = create_center(pos)

im = process_image(image, pos, centers, pred)

cv2.imwrite("test.jpg", im)


# print(V.size())

# with torch.no_grad():
# 	out = g.forward([V, A])
# [res, t] = out
# print(res.size())
# temp = res.numpy()
# print(temp)

# for i in temp:
# 	for j in i:
# 		if  j != 0:
# 			print(j)
