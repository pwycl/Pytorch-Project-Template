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

class SMTDataLoader:
	def __init__(self, config):
		self.config=config
		self.num_nodes=None

		if config.folds != None:
			self.set_loader(self.get_loader())

	def set_loader(self,loader):
		self.train_loader,self.val_loader,self.test_loader=loader

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
			num_nodes=min(int(num_nodes/len(dataset)), max_num_nodes)
			self.num_nodes=num_nodes
			indices=[]
			for i, data in enumerate(dataset):
				if data.num_nodes<=num_nodes:
					indices.append(i)
			dataset=dataset[torch.tensor(indices)]

			dataset.transform=T.ToDense(num_nodes)

		if dataset_div!=None:
			dataset=dataset.shuffle()[:len(dataset)//dataset_div]

		return dataset

	def get_k_fold_indices(self, folds,len_dataset):
		from sklearn.model_selection import StratifiedKFold
		skf=StratifiedKFold(folds,shuffle=True,random_state=12345)
		
		test_indices, train_indices=[],[]
		for _, idx in skf.split(torch.zeros(len_dataset),torch.zeros(len_dataset)):
			test_indices.append(torch.from_numpy(idx))

		val_indices=[test_indices[i-1] for i in range(folds)]

		for i in range(folds):
			train_mask=torch.ones(len_dataset, dtype=torch.uint8)
			train_mask[test_indices[i]]=0
			train_mask[val_indices[i]]=0
			train_indices.append(train_mask.nonzero().view(-1))

		return train_indices, val_indices, test_indices
		"""
		# test: assert isinstance(train_indices[i])==torch.tensor 
		

		"""

	# using generator to yield the train/val/test_loader 
	def k_fold_loader_generator(self,folds):
		dataset=self.get_dataset('SMT',sparse=self.config.sparse, dataset_div=self.config.dataset_div)
		train_indices,val_indices,test_indices=self.get_k_fold_indices(folds,len(dataset))

		for fold, (train_idx,val_idx,test_idx) in enumerate(
				zip(train_indices,val_indices,test_indices)):

			train_loader=DenseDataLoader(dataset[train_idx],self.config.batch_size,shuffle=True)
			val_loader=DenseDataLoader(dataset[val_idx],self.config.batch_size,shuffle=False)
			test_loader=DenseDataLoader(dataset[test_idx],self.config.batch_size,shuffle=False)

			yield train_loader,val_loader,test_loader




