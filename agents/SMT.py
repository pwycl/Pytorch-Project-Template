from agents.base import BaseAgent
from datasets.SMT import SMTDataLoader
from graphs.models.diff_pool import DiffPool

import tqdm
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from utils.misc import print_cuda_statistics

class SMTAgent(BaseAgent):
	def __init__(self,config):
		super().__init__(config)

		self.smt_loader=SMTDataLoader(config)
		self.loss=nn.NLLLoss()

		#set cuda flag
		self.is_cuda=torch.cuda.is_available()
		if self.is_cuda and not self.config.cuda:
			self.logger.info("WARNING: You have a CUDA device, so you should probably enable it!")
		self.cuda=self.is_cuda & self.config.cuda

		#set the manual seed for torch
		self.manual_seed=self.config.seed
		if self.cuda:
			torch.cuda.manual_seed(self.manual_seed)
			self.device=torch.device('cuda')
			# self.model=self.model.to(self.device)
			self.loss=self.loss.to(self.device)
			self.logger.info("Program will run on *****GPU-CUDA*****")
			print_cuda_statistics()
		else:
			self.device=torch.device("cpu")
			torch.manual_seed(self.manual_seed)
			self.logger.info("Program will run on *****CPU*****\n")

		# Model Loading from the latest checkpoint， if not found start from scratch. 
		self.load_checkpoint(self.config.checkpoint_file)
		# Summary Writer
		self.summary_writer=None

	def reset_parameters(self):
		self.model=DiffPool(self.smt_loader.train_loader.dataset,
							self.config.num_pools,self.config.hidden)
		if self.cuda:
			self.model.to(self.device)

		self.optimizer=optim.Adam(
			self.model.parameters(),
			lr=self.config.learning_rate,
			weight_decay=self.config.weight_decay
			)

		self.current_epoch=0
		self.current_iteration=0


	def load_checkpoint(self,file_name):
		"""
		Latest checkpoint loader
		:param file_name: name of the checkpoint file
		:return:
		"""
		pass

	def save_checkpoint(self,file_name="checkpoint.pth.tar", is_best=0):
		"""
		Checkpoint saver
		:param file_name: name of the checkpoint file
		:param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far.
		:return:
		"""
		pass

	def run_k_folds(self):
		for fold, dataloader in enumerate(
			self.smt_loader.k_fold_loader_generator(self.config.folds)):

			self.smt_loader.set_loader(dataloader)
			self.reset_parameters()
			self.logger.info('Folds {}:'.format(fold))
			self.train()


	def run(self):
		"""
		The main operator
		:return:
		"""
		try:
			if self.config.folds != None:
				self.run_k_folds()
			else:
				self.reset_parameters()
				self.train()
		except KeyboardInterrupt:
			self.logger.info("You have entered CTRL+c.. Wait to finalize")

	def train(self):
		"""
		Main training loop
		:return:
		"""
		for epoch in range(1, self.config.max_epoch+1):
			self.train_one_epoch()
			self.validate()
			self.current_epoch+=1

	def train_one_epoch(self):
		"""
		One epoch of training
		:return:
		"""
		self.model.train()
		for batch_idx, data in tqdm.tqdm(enumerate(self.smt_loader.train_loader)):
			data=data.to(self.device)
			self.optimizer.zero_grad()
			output=self.model(data)
			loss=self.loss(output,data.y.view(-1))
			loss.backward()
			self.optimizer.step()
			if batch_idx % self.config.log_interval==0:
				self.logger.info("Train Epoch: {}\tLoss:{:.6f}".format(
					self.current_epoch,loss.item()))
			self.current_iteration+=1

	def validate(self):
		"""
		One cycle of model validation
		:return:
		"""
		self.model.eval()
		test_loss=0
		correct=0
		with torch.no_grad():
			for data in self.smt_loader.val_loader:
				data=data.to(self.device)
				output=self.model(data)
				test_loss+=F.nll_loss(output,data.y.view(-1),reduction='sum').item()
				pred=output.max(1)[1]
				correct+=pred.eq(data.y.view_as(pred)).sum().item()

		test_loss/=len(self.smt_loader.val_loader.dataset)
		self.logger.info('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss,correct,len(self.smt_loader.val_loader.dataset),100.*correct/len(self.smt_loader.val_loader.dataset)))

	def finalize(self):
		"""
		Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
		:return:
		"""
		pass

