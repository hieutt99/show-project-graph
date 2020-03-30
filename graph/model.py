import torch
import torch.nn as nn
from graph.layers import GraphConvolution, GraphLinearEmbedding
from torch.nn import Module, Softmax, Dropout

class GCN(Module):
	def __init__(self,):
		super(GCN, self).__init__()
		self.size = 128
		self.in_features = 600
		self.l1 = GraphLinearEmbedding(self.in_features, self.size, with_bn = False, with_act_func = True)
		# self.dropout1 = Dropout(p = 0.5)
		
		self.gc1 = GraphConvolution(self.size, self.size*4, 4, with_bn = False, \
			with_act_func = True)
		# self.dropout2 = Dropout(p = 0.5)
		
		self.gc2 = GraphConvolution(self.size*4, self.size*8, 4, with_bn = False, \
			with_act_func = True)
		# self.dropout3 = Dropout(p = 0.5)
		# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~```
		self.gc6 = GraphConvolution(self.size*8, self.size*4, 4, with_bn = False, \
			with_act_func = True)

		self.gc7 = GraphConvolution(self.size*4, self.size, 4, with_bn = False, \
			with_act_func = True)
		# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		self.l3 = GraphLinearEmbedding(self.size, self.size*4, with_bn = False)

		self.gc3 = GraphConvolution(self.size*4, self.size*8, 4, with_bn = False, \
			with_act_func = True)

		self.gc4 = GraphConvolution(self.size*8, self.size*4, 4, with_bn = False, \
			with_act_func = True)

		self.gc5 = GraphConvolution(self.size*4, self.size, 4, with_bn = False, \
			with_act_func = True)

		self.l2 = GraphLinearEmbedding(self.size, 5, with_bn = False, with_act_func = True)
		
		self.softmax = Softmax(-1)

		self.criterion = torch.nn.CrossEntropyLoss()

	def forward(self, input):
		X = self.l1(input)

		X = self.gc1(X)
		X = self.gc2(X)
		X = self.gc6(X)
		X = self.gc7(X)
		
		X = self.l3(X)

		X = self.gc3(X)
		X = self.gc4(X)
		X = self.gc5(X)
		X = self.l2(X)

		
		return X

	def eval(self, input):
		X = self.l1(input)

		X = self.gc1(X)
		X = self.gc2(X)
		X = self.gc6(X)
		X = self.gc7(X)


		X = self.l3(X)

		X = self.gc3(X)
		X = self.gc4(X)
		X = self.gc5(X)
		X = self.l2(X)

		X[0] = self.softmax(X[0])
		
		return X

	def loss(self, output, target):
		return self.criterion(output, target)
