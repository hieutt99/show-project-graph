
from torch.nn import Module
import torch

class Explainer(Module):
	def __init__(self, model, dataset, train_idx, args):
		self.model = model
		self.model.eval()
		
		self.dataset = dataset

		self.train_idx = train_idx

		self.args = args
	

	def explain(self, node_idx, graph_idx = 0, graph_mode = False, unconstrained = False, model = 'exp'):
		'''
		explain a single node prediction
		'''

		device = torch.device('cuda:0')