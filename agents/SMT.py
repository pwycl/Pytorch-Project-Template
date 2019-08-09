import os.path as osp
import shutil

import tqdm
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from utils.misc import print_cuda_statistics
from agents.base import BaseAgent
from datasets.SMT import SMTDataLoader
from graphs.models.diff_pool import DiffPool


class SMTAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.smt_loader = SMTDataLoader(config)
        self.loss = nn.NLLLoss()

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info(
                "WARNING: You have a CUDA device, "
                + "so you should probably enable it!")
        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device('cuda')
            # self.model=self.model.to(self.device)
            self.loss = self.loss.to(self.device)
            self.logger.info("Program will run on *****GPU-CUDA*****")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

    def reset_parameters(self):
        self.model = DiffPool(self.smt_loader.train_loader.dataset,
                              self.config.num_pools, self.config.hidden)
        if self.cuda:
            self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.current_epoch = 0
        self.current_iteration = 0
        self.best_val_acc = 0

    def load_checkpoint(self, file_name='best_model_best.pth.tar'):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = osp.join(self.config.checkpoint_dir, file_name)
        try:
            self.logger.info(
                "Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
        except OSError:
            self.logger.info(
                "No checkpoint exists from '{}'. Skipping...".format(
                    self.config.checkpoint_dir))
            self.logger.info("**First time need to train**")
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(
                "Checkpoint loaded successfully from '{}'".format(
                    self.config.checkpoint_dir))

    def save_checkpoint(self, file_name, is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's
         accuracy is the best so far.
        :return:
        """
        state = {
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, self.config.checkpoint_dir+file_name)
        if is_best:
            shutil.copyfile(osp.join(self.config.checkpoint_dir, file_name),
                            osp.join(self.config.checkpoint_dir,
                                     str.join('best_', file_name)))

    def run_k_folds(self):
        for fold, dataloader in enumerate(
                self.smt_loader.k_fold_loader_generator(self.config.folds)):

            self.fold = fold
            self.smt_loader.set_loader(dataloader)
            self.reset_parameters()
            self.logger.info('Folds {}:'.format(fold))
            filename = '{}_checkpoint.pth.tar'.format(fold)
            self.train(checkpoint_filename=filename)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.folds is not None:
                self.run_k_folds()
            else:
                self.reset_parameters()
                self.train()
            self.logger.info(
                "\nBest validate set accuracy: {}".format(self.best_val_acc))
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+c.. Wait to finalize")

    def train(self, checkpoint_filename='checkpoint.pth.tar'):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch+1):
            self.train_one_epoch()
            val_acc = self.validate()
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.save_checkpoint(
                    file_name=checkpoint_filename,
                    is_best=is_best
                )
            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        for batch_idx, data in tqdm.tqdm(
                enumerate(self.smt_loader.train_loader)):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, data.y.view(-1))
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info("Train Epoch: {}\tLoss:{:.6f}".format(
                    self.current_epoch, loss.item()))
            self.current_iteration += 1

    def evaluate(self, data_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(
                    output,
                    data.y.view(-1), reduction='sum').item()
                pred = output.max(1)[1]
                correct += pred.eq(data.y.view_as(pred)).sum().item()

        test_loss /= len(data_loader.dataset)
        acc = correct/len(data_loader.dataset)
        log_info = ('\nAverage loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.0f}%)\n')
        log_info = log_info.format(
            test_loss, correct, len(data_loader.dataset),
            100.*acc
        )
        return log_info, acc

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        log_info, acc = self.evaluate(
            data_loader=self.smt_loader.val_loader
        )
        self.logger.info('Val set: '+log_info)
        return acc

    def test_checkpoint(self):
        self.reset_parameters()
        self.load_checkpoint()
        log_info, acc = self.evaluate(
            data_loader=self.smt_loader.test_loader
        )
        self.logger.info('Test set: '+log_info)
        return acc

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process,
        the operator and the data loader
        :return:
        """
        pass
