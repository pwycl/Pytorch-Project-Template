from math import ceil
import torch
from easydict import EasyDict
from torch_geometric.nn import dense_diff_pool

from graphs.models.diff_pool import DiffPool

config_dict={
	'batch_size':24,
	'sparse':False,
	'dataset_div':10,
	'num_nodes':5187,
	'channels':128,
	'num_layers':5,
	'hidden':64,
}
config=EasyDict(config_dict)
config.num_clusters=ceil(config.num_nodes*0.25)

# def test_pyg_diff_pool():
# 	batch_size,num_nodes,channels,num_clusters=(
# 					config.batch_size,config.num_nodes,config.channels,config.num_clusters)

# 	x=torch.randn((batch_size,num_nodes,channels))
# 	adj=torch.rand((batch_size,num_nodes,num_nodes))
# 	s=torch.randn((batch_size,num_nodes,num_clusters))
# 	mask=torch.randint(0,1,(batch_size,num_nodes),dtype=torch.uint8)

# 	x,adj,link_loss,ent_loss=dense_diff_pool(x,adj,s,mask)

# 	assert x.size()==(batch_size,num_clusters,channels)
# 	assert adj.size()==(batch_size,num_clusters,num_clusters)
# 	assert link_loss.item()>=0
# 	assert ent_loss.item()>=0

def test_Diff_Pool():
	from datasets.SMT import SMTDataLoader

	smt=SMTDataLoader(config)
	dataset=smt.get_dataset('SMT',sparse=config.sparse,dataset_div=config.dataset_div)
	model=DiffPool(dataset,config.num_layers,config.hidden)
	model.eval()
	for batch in smt.train_loader:
		assert list(batch.x.size())[:]==[config.batch_size,smt.num_nodes,dataset[0].num_features]
		assert list(batch.adj.size())[1:]==[smt.num_nodes,smt.num_nodes]
		assert list(batch.mask.size())[1:]==[smt.num_nodes]
		with torch.no_grad():
			out=model(batch)
			assert list(out.size())[1:]==[dataset.num_classes]


