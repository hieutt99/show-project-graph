import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from keras.preprocessing.text import Tokenizer
# from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import json
import string as S
from warnings import filterwarnings
filterwarnings('ignore')
# =============================================================================
def clean_word(string): 
	'''
	remove punctuations, special character and numbers. join with 1 space
	'''
	for x in string.lower(): 
		if x.isdigit():
			string = string.replace(x, " numberfield ")
		if x in S.punctuation or (not x.isalpha() and x!=" "): 
			string = string.replace(x, "") 
	return ' '.join((string.lower()).split())

def read_text(link):
	'''
	read text line by line from files with link
	'''
	f = open(link,"r")
	count = 0
	page = []
	while True:
		ls = []
		count+=1 
		line = f.readline().rstrip('\n').strip()
		
		if not line:
			break
		ls = line.split(',', maxsplit = 8)        
		
		for i,item in enumerate(ls[:8]):
			ls[i] = int(item)
		page.append(ls)
	f.close()
	return page

def get_page(name):
	'''
	return processed textlines list, position of each lines and labels of each line
	by link of page
	'''
	link = "./data/task1train_pro/"+name+".txt"
	label_link = "./data/task2train_pro/"+name+".json"

	with open(label_link, "r") as fp:
		label_dict = json.load(fp)
	# print(label_dict)
	page = read_text(link)
	
	slice = []
	position = []
	label = []
	
	for item in page:
		textline = item[-1]
		check = 0
		for key, val in label_dict.items():
			if val == textline or val in textline or textline in val and len(textline) > 2:
				label.append(key)
				check = 1 
				break
		if check == 0:
			label.append("_")

		slice.append(clean_word(textline))
		position.append(item[:-1])

	return slice, position, label
# ==================================================================================
def get_page_demo(file_name):
	'''
	return processed textlines list, position of each lines 
	'''
	link = "./part1_output/"+file_name

	page = read_text(link)
	
	slice = []
	position = []
	
	for item in page:
		textline = item[-1]

		
		slice.append(clean_word(textline))
		position.append(item[:-1])

	return slice, position
# =========================================================================================
def create_center(pos):
	'''
	create centers from positions list and return in form of numpy array
	'''
	center = []
	for item in pos:
		y = int((item[3]+item[5])/2)
		x = int((item[0]+item[2])/2)
   
		center.append([x,y])

	center = np.array(center)
	
	center = np.hstack((center, np.arange(center.shape[0]).reshape(center.shape[0],1)))
	return center

def create_adj_matrix(center, pos, n_edges = None):
	'''
	return adjacency matrix from centers array and positions array
	'''
	'''
	positions [N * 9]  positions[-1] is index in the original 
	center if ndim != 3 should be concatenated with index in the original
	'''
	'''
	positions [N * 9]  positions[-1] is index in the original 
	center if ndim != 3 should be concatenated with index in the original
	'''
	'''
	Init adj array in shape of L*N*N with L = 4 hopefully fixed
	'''
	array = np.zeros((4, center.shape[0], center.shape[0]))

	if center.ndim != 3 :
		centers = np.hstack((center, np.arange(center.shape[0]).reshape(center.shape[0],1)))
	else :
		centers = center
	# Cat the indexes into boxes
	positions = np.hstack((np.array(pos), np.arange(len(pos)).reshape(len(pos),1)))

	# top down 
	temp = positions[positions[:, 5].argsort()]

	for i, _ in enumerate(temp[:-1]):
		count = 0
		check = 0
		for j in range(i+1, temp.shape[0]):
			# if count == 2 and check == 0:
			# 	break
			if (temp[i][6]<=temp[j][2] and temp[j][2]<=temp[i][4]) or\
				(temp[i][6]<=temp[j][0] and temp[j][0]<=temp[i][4]) or\
				(temp[i][6]>=temp[j][0] and temp[i][4]<=temp[j][2]) or\
				(temp[i][0]<=temp[j][0] and temp[j][2]<=temp[i][4]) :
				start = temp[i][-1]
				stop = temp[j][-1]
				array[0,start,stop] = 1
				break
			count+=1
	# up
	temp = positions[positions[:, 1].argsort()][::-1]

	for i, _ in enumerate(temp[:-1]):
		count = 0
		check = 0
		for j in range(i+1, temp.shape[0]):
			# if count == 2 and check == 0:
			# 	break
			if (temp[i][6]<=temp[j][2] and temp[j][2]<=temp[i][4]) or\
				(temp[i][6]<=temp[j][0] and temp[j][0]<=temp[i][4]) or\
				(temp[i][6]>=temp[j][0] and temp[i][4]<=temp[j][2]) or\
				(temp[i][0]<=temp[j][0] and temp[j][2]<=temp[i][4]) :
				start = temp[i][-1]
				stop = temp[j][-1]
				array[1,start,stop] = 1
				break	
			count+=1
	# left - right
	temp = positions[positions[:, 1].argsort()]

	for i, _ in enumerate(temp[:-1]):
		count = 0
		check = 0
		for j in range(i+1, temp.shape[0]):
			# if count == 2 and check == 0:
			# 	break
			if (temp[i][7]<=temp[j][7] and temp[j][1]<=temp[i][7]) or\
				(temp[i][1]<=temp[j][7] and temp[j][1]<=temp[i][1]) or\
				(temp[i][1]<=temp[j][1] and temp[i][7]>=temp[j][7]) or\
				(temp[i][1]>=temp[j][1] and temp[j][7]<=temp[i][7]) :
				start = temp[i][-1]
				stop = temp[j][-1]
				array[2,start,stop] = 1
				array[3,stop,start] = 1
				break	
			count+=1

	return array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# ========================================================================
class ToTensor(object):
	def __call__(self, sample):
		'''
		In a tensor only have 1 V array N*C , 1 A tensor K*N*N and 1 label array N
		'''
		[V, A], Y = sample['VA'], sample['label']
		# print(Y.shape)
		res =  [[torch.from_numpy(V).type(torch.FloatTensor),torch.from_numpy(A).type(torch.FloatTensor)], torch.from_numpy(Y).type(torch.LongTensor)]
		return res
class data_process(Dataset):
	def __init__(self, data_list, transform = None):
		self.data_list = data_list
		self.transform = transform
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`


		with open('./bow/docs.json','r') as fp:
			self.docs = json.load(fp)
		self.vectorizer = CountVectorizer(max_features = 600)
		self.vectorizer.fit(self.docs)
		self.label_list = ['_','company','date','address','total']

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		s, pos, label = get_page(self.data_list[idx])
		# print(label)

		# V = self.tokenizer.texts_to_matrix(s, mode = 'count')

		V = self.vectorizer.transform(s).toarray()
		# V = np.hstack((V, np.array(pos)))
		centers = create_center(pos)
		A = create_adj_matrix(centers, pos)
		num_label = np.array([self.label_list.index(element) for element in label])

		sample = {'VA':[V, A], 'label':num_label}
		if self.transform :
			sample = self.transform(sample)
		return sample
class DemoToTensor(object):
	def __call__(self, sample):
		'''
		In a tensor only have 1 V array N*C , 1 A tensor K*N*N and no label
		'''
		[V, A] = sample['VA']
		# print(Y.shape)
		res =  [torch.from_numpy(V).type(torch.FloatTensor),torch.from_numpy(A).type(torch.FloatTensor)]
		return res
class demo_preprocess(Dataset):
	def __init__(self, data_list, transform = DemoToTensor()):
		self.data_list = data_list
		self.transform = transform
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`


		with open('./bow/docs.json','r') as fp:
			self.docs = json.load(fp)
		self.vectorizer = CountVectorizer(max_features = 600)
		self.vectorizer.fit(self.docs)

	def __len__(self):
		return len(self.data_list)


	def __getitem__(self, idx):
		s, pos = get_page_demo(self.data_list[idx])

		V = self.vectorizer.transform(s).toarray()

		centers = create_center(pos)
		A = create_adj_matrix(centers, pos)
		# for i in V:
		# 	for j in i:
		# 		print(j, end = "")
		# 	print()


		sample = {'VA':[V, A]}
		if self.transform :
			sample = self.transform(sample)
		return sample