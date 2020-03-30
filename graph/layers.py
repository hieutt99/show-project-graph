import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Module, ParameterList, BatchNorm1d, ReLU, Softmax, LayerNorm
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class GraphConvolution(Module):
	def __init__(self, in_features, out_features, L, with_bn = True, with_act_func = True):
		super(GraphConvolution, self).__init__()
		'''
		W : [in_f * L, out_f]
		b : [out_f]
		'''
		self.device = torch.device('cpu')

		self.params = Parameter(nn.init.xavier_uniform_(torch.FloatTensor(in_features * \
			(L+1), out_features)))
		self.bias = Parameter(nn.init.normal_(torch.FloatTensor(out_features), mean = 0.001, std = 0.0005))

		self.with_bn = with_bn
		self.with_act_func = with_act_func

	def forward(self, input):
		'''
		V : [N, in_f]
		A : [L, N, N]

		V_out = [N, out_f]
		'''
		[V, A] = input

		# Identity cat
		I = torch.eye(A.size(1)).view(1,A.size(1),A.size(1)).to(self.device)
		A_cat = torch.cat([I,A], dim = 0)
		# Reshape
		A_reshape = A_cat.view(A_cat.size(0)*A_cat.size(1), A_cat.size(1))
		X_hat = torch.mm(A_reshape, V)

		X_hat = X_hat.view(V.size(0), A_cat.size(0)* V.size(1))
		
		res = torch.mm(X_hat.float(), self.params) + self.bias
		# res = torch.mm(X_hat.float(), self.params)
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# conditional output
		if self.with_bn:
			bn = BatchNorm1d(res.size(1), affine = False).cuda()
			# norm = LayerNorm(res.size(1))
			res = bn(res)
			# res = norm(res)
		if self.with_act_func:
			relu = ReLU()
			res = relu(res)
		return [res,A]


class GraphLinearEmbedding(Module):
	def __init__(self, in_features, out_features,
				with_bn = False, with_act_func = True):
		super(GraphLinearEmbedding, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.with_bn = with_bn
		self.with_act_func = with_act_func

		'''
		W : [in_f, out_f]
		b : [out_f]
		'''
		self.params = Parameter(nn.init.xavier_uniform_(torch.FloatTensor(self.in_features,self.out_features)))
		self.bias = Parameter(nn.init.normal_(torch.FloatTensor(self.out_features), mean = 0.001, std = 0.0005))

	def forward(self, input):
		'''
		V : [N, in_f]

		V_out = [N, out_f]
		'''
		[V, A] = input
		V = torch.mm(V, self.params) + self.bias
		# V = torch.mm(V, self.params)
		if self.with_bn:
			bn = BatchNorm1d(V.size(1), affine = False).cuda()
			V = bn(V)
			# norm = LayerNorm(V.size(1))
			# V = norm(V)
		if self.with_act_func:
			relu = ReLU()
			V = relu(V)
		return [V,A]
