import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler, get_worker_info
import numpy as np
from metrics import get_metrics_multiclass, get_metrics
import os 
import pandas as pd
import pickle5 as pickle 
from net import MlpMixer
from CNN3D import C3D
import torch.nn as nn
from dataset import MixerDataset, IterativeDataset
import torch.distributed as dist


def get_task(args):
    return MixerTask(args)

def load_task(ckpt_path, **kwargs):
    return MixerTask.load_from_checkpoint(ckpt_path, **kwargs)

class MixerTask(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        time = args.time
        people = args.people
        joints = args.joints
        channels = 3
        num_classes = args.num_classes
        self.num_classes = num_classes
        self.model = C3D(time, people, joints, channels, num_classes)
        self.val_metrics = []
        self.test_metrics = []
        self.loss = nn.BCELoss()
    
    def forward(self, x):
        return self.model(x.to(torch.float32)) #Necessary to prevent conversion issues ) 

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        x, y = batch["embedding_seq"], batch["label"]
        logits = self.forward(x) 
        loss = self.loss(logits, y.long())
        self.log("train_loss", loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        x, y = batch["embedding_seq"], batch["label"].to(torch.float32)
        logits = self.forward(x) 
        loss = self.loss(logits, y.long())
        probs = torch.softmax(logits, dim=1)
        metrics = {'labels': y, 'logits': logits, 'probs': probs, 'val_loss': loss}
        self.val_metrics.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        """
        Aggregate and return the validation metrics
        Args:
        outputs: A list of dictionaries of metrics from `validation_step()'
        Returns: None
        Returns:
            A dictionary of loss and metrics, with:
                val_loss (required): validation_loss
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        print('validation epoch end')
        outputs = self.val_metrics
        avg_loss = torch.stack([batch['val_loss'] for batch in outputs]).mean()
        labels = torch.cat([batch['labels'] for batch in outputs])
        probs = torch.cat([batch['probs'] for batch in outputs])
        metrics_strategy = self.hparams['metrics_strategy']

        #Log Val Accuracy and Loss
        self.log("val_loss", avg_loss.item())

        #Log Val Metrics
        if self.num_classes == 2:
            metrics = get_metrics(labels, probs)
        else:
            metrics = get_metrics_multiclass(labels, probs, metrics_strategy) 
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value)
            
        self.val_metrics.clear()

    def test_step(self, batch, batch_nb):
        x, y = batch["embedding_seq"], batch["label"].to(torch.float32)
        logits = self.forward(x) 
        loss = self.loss(logits, y.long())
        probs = torch.softmax(logits, dim=1)
        metrics = {'labels': y, 'logits': logits, 'probs': probs, 'val_loss': loss}
        self.test_metrics.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        
        outputs = self.test_metrics
        avg_loss = torch.stack([batch['val_loss'] for batch in outputs]).mean()
        labels = torch.cat([batch['labels'] for batch in outputs])
        logits = torch.cat([batch['logits'] for batch in outputs])
        probs = torch.cat([batch['probs'] for batch in outputs])
        metrics_strategy = self.hparams['metrics_strategy']

        #Log Test Accuracy and Loss
        self.log("test_loss", avg_loss)

        #Log Test Metrics
        if self.num_classes == 2:
            metrics = get_metrics(labels, probs)
        else:
            metrics = get_metrics_multiclass(labels, probs, metrics_strategy) 
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)
        metrics['default'] = metrics['auprc']

        self.test_metrics.clear()
        
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        learn_rate = self.hparams['learn_rate']
        if self.hparams['optimizer'] == 'Adam':
            weight_decay = self.hparams.get('weight_decay', 0)  
            return [torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=weight_decay)]
        elif self.hparams['optimizer'] == 'AdamW':
            weight_decay = self.hparams.get('weight_decay', 1e-5) 
            return [torch.optim.AdamW(self.parameters(), lr=learn_rate, weight_decay=weight_decay)]
        else:
            return [torch.optim.SGD(self.parameters(), lr=learn_rate, momentum=0.9)]

    def train_dataloader(self):
        oversample = self.hparams['oversample']
        dataset_path = self.hparams.get('dataset_path', "")
        dataset = IterativeDataset(dataset_path, 'train')  
        train_sampler = DistributedSampler(dataset, shuffle=True)  # Ensures correct data distribution

        #Create oversampling weights 
        if oversample:
            ref_dataset = read_pickle(dataset_path)['train']
            counts = {} 
            for example in ref_dataset: 
                label = str(example[1])
                if label not in counts:
                    counts[label] = 1 
                else:
                    counts[label] += 1 
            weights = [] 
            for example in ref_dataset: 
                label = str(example[1])
                weight = 1 / counts[label] 
                weights.append(weight) 
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights)
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
            
        if dist.is_initialized():
            train_sampler = DistributedSampler(dataset, shuffle=True)  # Ensures correct data distribution
            drop = True
        else: 
            train_sampler = None
            drop = False
        return DataLoader(dataset, shuffle=shuffle, batch_size=self.hparams['batch_size'], num_workers=8, sampler=train_sampler, drop_last=drop, worker_init_fn=self.worker_init_fn)

    def val_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        dataset = IterativeDataset(dataset_path, 'valid') 
        if dist.is_initialized():
            train_sampler = DistributedSampler(dataset, shuffle=True)  # Ensures correct data distribution
            drop = True
        else: 
            train_sampler = None
            drop = False
        return DataLoader(dataset, shuffle=False, batch_size=self.hparams['batch_size'], num_workers=8, drop_last=drop, sampler= train_sampler, worker_init_fn=self.worker_init_fn)

    def test_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        dataset = IterativeDataset(dataset_path, 'test') 
        if dist.is_initilized():
            train_sampler = DistributedSampler(dataset, shuffle=True)  # Ensures correct data distribution
            drop = True
        else:
            drop = False
            train_sampler = None
        return DataLoader(dataset, shuffle = False, batch_size = self.hparams['batch_size'], drop_last=drop, sampler= train_sampler, num_workers = 8, worker_init_fn=self.worker_init_fn) #fix num workers 

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        if worker_info is not None:
            torch.manual_seed(worker_info.seed)  # Ensures different seeds per worker
            
#HELPER FUNCTIONS 
def write_pickle(data_object, path):
    with open(path, 'wb') as handle:
        pickle.dump(data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')
