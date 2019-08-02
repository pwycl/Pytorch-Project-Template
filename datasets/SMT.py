from __future__ import division

import sys
import random
import os.path as osp
import shutil

import pytest
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, DenseDataLoader, DataListLoader
import torch_geometric.transforms as T

#####test#####

# from easydict import EasyDict
# config_dict={
# 	'batch_size': 24,
# 	'sparse': True,
# 	'dataset_div': 10,
# }
# config=EasyDict(config_dict)
#####test#####

class SMTDataLoader:
	def __init__(self, config):
		self.config=config

		self.num_nodes=None
		self.train_loader, self.val_loader, self.test_loader=self.get_loader()

	def get_loader(self):		#paras config->self.config
		dataset=self.get_dataset('SMT', sparse=self.config.sparse, dataset_div=self.config.dataset_div)
		n=(len(dataset)+9)//10
		test_dataset=dataset[:n]
		val_dataset=dataset[n:2*n]
		train_dataset=dataset[2*n:]

		train_loader=DenseDataLoader(train_dataset,batch_size=self.config.batch_size)
		val_loader=DenseDataLoader(val_dataset,batch_size=self.config.batch_size)
		test_loader=DenseDataLoader(test_dataset,batch_size=self.config.batch_size)

		return train_loader, val_loader, test_loader

	def get_dataset(self, name, sparse=True, dataset_div=None):
		path=osp.join(osp.dirname(osp.realpath(__file__)), '..','data',name)
		try:
			shutil.copytree('../input/smt',path)
		except FileExistsError as e:
			print(e)

		dataset=TUDataset(path,name,use_node_attr=True)
		dataset.data.edge_attr=None

		if not sparse:
			num_nodes=max_num_nodes=0
			for data in dataset:
				num_nodes+=data.num_nodes
				max_num_nodes=max(data.num_nodes,max_num_nodes)

			#Filter out a few really large graphs in order to apply DiffPool
			self.num_nodes=min(int(num_nodes/len(dataset)*5), max_num_nodes)
			indices=[]
			for i, data in enumerate(dataset):
				if data.num_nodes<=num_nodes:
					indices.append(i)
			dataset=dataset[torch.tensor(indices)]

			dataset.transform=T.ToDense(num_nodes)

		if dataset_div!=None:
			dataset=dataset.shuffle()[:len(dataset)//dataset_div]

		return dataset







