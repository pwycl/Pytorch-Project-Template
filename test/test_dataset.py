import sys
import os.path as osp
import shutil
import random
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import pytest

from datasets.SMT import SMTDataLoader

# @pytest.mark.unfinished
def test_baseDataset_Loader(filted_dataset=None):
	root=osp.join('/','tmp', str(random.randrange(sys.maxsize)))
	shutil.copytree('../input/smt', root)
	dataset=TUDataset(root,'SMT')

	assert len(dataset)==2688
	assert dataset.num_features==20
	assert dataset.num_classes==2
	assert dataset.__repr__()=='SMT(2688)'
	assert dataset[0].keys==['x', 'edge_index', 'y']   #==len(data.keys)  
	assert len(dataset.shuffle())==2688

	loader=DataLoader(dataset,batch_size=len(dataset))
	assert loader.dataset.__repr__()=='SMT(2688)'
	for batch in loader:
		assert batch.num_graphs==2688
		assert batch.num_nodes==sum([data.num_nodes for data in dataset])  #2788794
		assert batch.num_edges==sum([data.num_edges for data in dataset])  #13347768
		assert batch.keys==['x','edge_index','y','batch']

	num_nodes=sum(dataset.data.num_nodes)
	max_num_nodes=max(dataset.data.num_nodes)
	num_nodes=min(int(num_nodes/len(dataset)*5),max_num_nodes)

	assert num_nodes==5187
	assert max_num_nodes==34623

	indices=[]
	for i,data in enumerate(dataset):
		if data.num_nodes<num_nodes:
			indices.append(i)

	if filted_dataset==None:
		filted_dataset=dataset[torch.tensor(indices)]
		filted_dataset.transform=T.ToDense(num_nodes)  #add 'adj' attribute

	assert ('adj' in dataset[0])==False
	assert ('adj' in filted_dataset[0])==True


# @pytest.mark.finished
def test_SMT():
	from easydict import EasyDict
	config_dict={
		'batch_size': 24,
		'sparse': False,
		'dataset_div': 10,
	}
	config=EasyDict(config_dict)

	smt=SMTDataLoader(config)
	dataset=smt.get_dataset('SMT', 
		sparse=config.sparse, dataset_div=config.dataset_div)

	assert max(dataset.data.num_nodes) <= 5187

	test_baseDataset_Loader(dataset)  # expensive mem cost
	# assert ('adj' in dataset[0])==True

	for batch in smt.train_loader:
		assert batch.keys==['x', 'y', 'adj', 'mask']
		assert list(batch.x.size())==[config.batch_size,smt.num_nodes, dataset.data.num_features] # [24, 5187, 20]
		assert list(batch.adj.size())==[config.batch_size,smt.num_nodes,smt.num_nodes] # [24,5187, 5187]
		assert list(batch.mask.size())==[config.batch_size,smt.num_nodes]
		assert list(batch.y.size())==[config.batch_size,1]

	for batch in smt.val_loader:
		assert batch.keys==['x', 'y', 'adj', 'mask']
		assert list(batch.x.size())==[config.batch_size,smt.num_nodes, dataset.data.num_features] # [24, 5187, 20]
		assert list(batch.adj.size())==[config.batch_size,smt.num_nodes,smt.num_nodes] # [24,5187, 5187]
		assert list(batch.mask.size())==[config.batch_size,smt.num_nodes]
		assert list(batch.y.size())==[config.batch_size,1]

	for batch in smt.test_loader:
			assert batch.keys==['x', 'y', 'adj', 'mask']
			assert list(batch.x.size())==[config.batch_size,smt.num_nodes, dataset.data.num_features] # [24, 5187, 20]
			assert list(batch.adj.size())==[config.batch_size,smt.num_nodes,smt.num_nodes] # [24,5187, 5187]
			assert list(batch.mask.size())==[config.batch_size,smt.num_nodes]
			assert list(batch.y.size())==[config.batch_size,1]



