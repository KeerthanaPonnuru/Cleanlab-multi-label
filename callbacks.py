import time
import json
from typing import Any
from omegaconf import OmegaConf

from lightning.pytorch.core import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only,rank_zero_info
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning.pytorch import Trainer
import lightning.pytorch as pl
import torch

from einops import rearrange

import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from sklearn.metrics import classification_report
from functools import partial
import os


class PredictLogger(Callback):
    def __init__(self,logdir):
        super().__init__()
        self.logdir = logdir
        self.probs = []
        self.preds = []
        self.paths = []
    
    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        probs = outputs.sigmoid()
        probs = probs.detach().cpu().numpy()
        self.probs.extend(probs)

        preds = np.where(probs > 0.5, 1, 0)
        self.preds.extend(preds)

        paths = batch['path']
        self.paths.extend(paths)
    
    def reset(self):
        self.probs = [];self.preds = []; self.paths = []

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.probs = np.array(self.probs)
        self.preds = np.array(self.preds)
        self.paths = np.array(self.paths)
        df_probs = pd.DataFrame(data = np.concatenate((self.paths[:,None],self.probs),axis=1), columns = ['path']+pl_module.AUs).sort_values(by=['path'])
        df_probs.to_csv(os.path.join(self.logdir,'Pred_prob.csv'),index=False)
        
        df_preds = pd.DataFrame(data = np.concatenate((self.paths[:,None],self.preds),axis=1), columns = ['path']+pl_module.AUs).sort_values(by=['path'])
        df_preds.to_csv(os.path.join(self.logdir,'Pred_pred.csv'),index=False)
        self.reset()
        
        
    '''def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.logger.experiment[f'prediction/probs'].upload(os.path.join(self.logdir,'Predcition_probs.csv'))
        pl_module.logger.experiment[f'prediction/preds'].upload(os.path.join(self.logdir,'Predcition_preds.csv'))'''


class MetricLogger(Callback):
    def __init__(self,logdir):
        super().__init__()
        self.logdir = logdir
        self.train_probs = []
        self.train_labels = []
        self.train_preds = []
        # self.train_paths = []
        # self.train_dataset = []

        self.val_probs = []
        self.val_labels = []
        self.val_preds = []
        # self.val_paths = []
        # self.val_dataset = []

        self.test_probs = []
        self.test_labels = []
        self.test_preds = []
        self.test_paths = []
        self.test_dataset = []
        # self.dataset_AUs = {'BP4D':['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU9', 'AU10','AU12', 'AU14', 'AU15', 'AU17','AU20', 'AU23', 'AU24', 'AU27'],
        #                     'DISFA':['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU17', 'AU20', 'AU25', 'AU26'],
        #                     'UNBC':[ 'AU4', 'AU6', 'AU7', 'AU9', 'AU10','AU12', 'AU15', 'AU20', 'AU25','AU26','AU27','AU43'],
        #                     'ICU':[ 'AU4', 'AU6', 'AU7', 'AU9', 'AU12',  'AU20', 'AU24', 'AU25','AU26','AU27','AU43'],
        #                     'ICUOLD':['AU4', 'AU6', 'AU7', 'AU9', 'AU12',  'AU20', 'AU24', 'AU25','AU26','AU27','AU43']}


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        prob = outputs['logits']
        prob = prob.sigmoid()
        prob = pl_module.all_gather(prob)
        prob = rearrange(prob, 'p b c -> (p b) c').detach().cpu().numpy()
        self.train_probs.extend(prob)

        label = batch["aus"]
        label = pl_module.all_gather(label)
        label = rearrange(label, 'p b c -> (p b) c').detach().cpu().numpy()
        self.train_labels.extend(label)

        preds = np.where(prob > 0.5, 1, 0)
        self.train_preds.extend(preds)

        # paths = batch["file_path_"]
        # paths = pl_module.all_gather(paths)
        # self.train_paths.extend(paths)

        # dataset = batch["dataset"]
        # dataset = pl_module.all_gather_object(dataset)
        # self.train_dataset.extend(dataset)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx =0) -> None:
        if not trainer.sanity_checking:
            prob = outputs['logits']
            prob = prob.sigmoid()
            prob = pl_module.all_gather(prob)
            prob = rearrange(prob, 'p b c -> (p b) c').detach().cpu().numpy()
            self.val_probs.extend(prob)

            label = batch["aus"]
            label = pl_module.all_gather(label)
            label = rearrange(label, 'p b c -> (p b) c').detach().cpu().numpy()
            self.val_labels.extend(label)

            preds = np.where(prob > 0.5, 1, 0)
            self.val_preds.extend(preds)

            # paths = batch["file_path_"]
            # paths = pl_module.all_gather(paths)
            # self.val_paths.extend(paths)

            # dataset = batch["dataset"]
            # dataset = pl_module.all_gather(dataset)
            # self.val_dataset.extend(dataset)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx =0) -> None:
        prob = outputs['logits']
        prob = prob.sigmoid()
        prob = pl_module.all_gather(prob)
        if len(prob.shape) == 3:
            prob = rearrange(prob, 'p b c -> (p b) c').detach().cpu().numpy()
        else:
            prob = prob.detach().cpu().numpy()
        self.test_probs.extend(prob)

        label = batch["aus"]
        label = pl_module.all_gather(label)
        if len(label.shape) == 3:
            label = rearrange(label, 'p b c -> (p b) c').detach().cpu().numpy()
        else:
            label = label.detach().cpu().numpy()
        self.test_labels.extend(label)

        preds = np.where(prob > 0.5, 1, 0)
        self.test_preds.extend(preds)

        paths = batch["file_path_"]
        self.test_paths.extend(paths)

        dataset = batch["dataset"]
        self.test_dataset.extend(dataset)

        
    def reset(self,split='train'):
        if split == 'train':
            self.train_probs = []
            self.train_labels = []
            self.train_preds = []
            self.train_paths = []
        else:
            self.val_probs = []
            self.val_labels = []
            self.val_preds = []
            self.val_paths = []


    def log_stats(self, trainer, stats, split='train'):
        for k,v in stats.items():
            if isinstance(v, dict):
                self.log_stats(trainer, v,split = split+'/'+str(k))
            else:
                trainer.logger.log_metrics({split+'/'+str(k): v}, step=trainer.current_epoch)

    def compute_metrics(self, label, preds, AUs):
        report = defaultdict(dict)
        roundn = partial(round,ndigits=2)
        for true,pred,AU in zip(label.T,preds.T,AUs):

            true_inds = np.where(true != -1)[0]
            true = true[true_inds]
            pred = pred[true_inds]

            reports = classification_report(true, pred,output_dict=True,labels=[0,1],zero_division=1)
            try:
                precision = reports['1']['precision']
                accuracy = reports['accuracy']
                recall = reports['1']['recall']
                specificity = reports['0']['recall']
                f1score = reports['1']['f1-score']
                support = reports['1']['support']

                report["f1"][AU] = roundn(f1score)
                report["precision"][AU] = roundn(precision)
                report["acc"][AU] = roundn(accuracy)
                report["recall"][AU] = roundn(recall)
                report["specificity"][AU] = roundn(specificity)
                report['support'][AU] = support
            except:
                pass    
        for key in list(report.keys()):
            if key != 'support':
                report[key]['Mean'] = roundn(np.mean([v for k,v in report[key].items()]))

        return report

    def compute_step(self,  pl_module, split='train'):
        #dataset_report = defaultdict(dict)
        if split == 'train':
            report = self.compute_metrics(np.array(self.train_labels), np.array(self.train_preds), pl_module.AUs)
        elif split == 'val':
            report = self.compute_metrics(np.array(self.val_labels), np.array(self.val_preds), pl_module.AUs)
        elif split == 'test':
            report = self.compute_metrics(np.array(self.test_labels), np.array(self.test_preds), pl_module.AUs)
        else:
            raise ValueError(f"Split {split} not recognized.")
        # dataset_report['all'] = report
        # datasets = np.unique(self.train_dataset)
        # if len(datasets) > 1:
        #     for dataset in datasets:
        #         idx = np.where(np.array(self.train_dataset) == dataset)[0]
        #         report = self.compute_metrics(np.array(self.train_labels)[idx], np.array(self.train_preds)[idx], pl_module.AUs,dataset=dataset)
        #         dataset_report[dataset] = report
        #save report as json
        epoch = str(pl_module.trainer.current_epoch).zfill(3)
        path = os.path.join(self.logdir,'reports',epoch)
        os.makedirs(path,exist_ok=True)
        with open(os.path.join(path,f'{split}.json'),'w') as f:
            json.dump(report,f)
        return report

    def make_dataframes(self,trainer,pl_module,aus,split='train'):
        if split == 'train':
            train_dataset = np.asarray(self.train_dataset)[:,None]
            train_paths = np.asarray(self.train_paths)[:,None]
            train_probs = np.asarray(self.train_probs)
            train_labels = np.asarray(self.train_labels)
            train_preds = np.asarray(self.train_preds)
            df_probs = pd.DataFrame(data = np.concatenate((train_dataset,train_paths,train_probs),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
            df_labels = pd.DataFrame(data = np.concatenate((train_dataset,train_paths,train_labels),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
            df_preds = pd.DataFrame(data = np.concatenate((train_dataset,train_paths,train_preds),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
        elif split == 'val':
            val_dataset = np.asarray(self.val_dataset)[:,None]
            val_paths = np.asarray(self.val_paths)[:,None]
            val_probs = np.asarray(self.val_probs)
            val_labels = np.asarray(self.val_labels)
            val_preds = np.asarray(self.val_preds)
            df_probs = pd.DataFrame(data = np.concatenate((val_dataset,val_paths,val_probs),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
            df_labels = pd.DataFrame(data = np.concatenate((val_dataset,val_paths,val_labels),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
            df_preds = pd.DataFrame(data = np.concatenate((val_dataset,val_paths,val_preds),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
        elif split == 'test':
            test_dataset = np.asarray(self.test_dataset)[:,None]
            test_paths = np.asarray(self.test_paths)[:,None]
            test_probs = np.asarray(self.test_probs)
            test_labels = np.asarray(self.test_labels)
            test_preds = np.asarray(self.test_preds)
            df_probs = pd.DataFrame(data = np.concatenate((test_dataset,test_paths,test_probs),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
            df_labels = pd.DataFrame(data = np.concatenate((test_dataset,test_paths,test_labels),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
            df_preds = pd.DataFrame(data = np.concatenate((test_dataset,test_paths,test_preds),axis=1), columns = ['dataset','path']+aus).sort_values(by=['dataset','path'])
        else:
            raise ValueError(f"Split {split} not recognized.")
        epoch = str(trainer.current_epoch).zfill(3)
        path = os.path.join(self.logdir,'dataframes',epoch)
        os.makedirs(path,exist_ok=True)
        df_probs.to_csv(os.path.join(path,'probs.csv'),index=False)
        df_labels.to_csv(os.path.join(path,'labels.csv'),index=False)
        df_preds.to_csv(os.path.join(path,'preds.csv'),index=False)

        pl_module.logger.experiment[f'{split}/{epoch}/probs'].upload(os.path.join(path,'probs.csv'))
        pl_module.logger.experiment[f'{split}/{epoch}/labels'].upload(os.path.join(path,'labels.csv'))
        pl_module.logger.experiment[f'{split}/{epoch}/preds'].upload(os.path.join(path,'preds.csv'))
        

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            val_report = self.compute_step(pl_module,split='val')
            #epoch = 'epoch'+str(trainer.current_epoch).zfill(3)
            self.log_stats(trainer, val_report, split=f'val')
            # if trainer.current_epoch == trainer.max_epochs-1:
            #     self.make_dataframes(trainer,pl_module,pl_module.AUs,split='val')
            self.reset('val')

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        train_report = self.compute_step(pl_module,split='train')
        #epoch = 'epoch'+str(trainer.current_epoch).zfill(3)
        self.log_stats(trainer, train_report, split=f'train')
        # if trainer.current_epoch == trainer.max_epochs-1:
        #     self.make_dataframes(trainer,pl_module,pl_module.AUs,split='train')   
        self.reset('train')
    
    @rank_zero_only
    def on_test_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        test_report = self.compute_step(pl_module,split='test')
        self.log_stats(trainer, test_report, split='test')
        self.make_dataframes(trainer,pl_module,pl_module.AUs,split='test')   
        self.reset('test')
        


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.now = now
        self.resume = resume
        self.config = config
        self.lightning_config = lightning_config
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
    
    def on_exception(self, trainer, pl_module,exception):
        if trainer.global_rank == 0:
            print(exception)
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path) 
            pl_module.logger.experiment[f'checkpoints/last.ckpt'].upload(os.path.join(self.logdir,'checkpoints','last.ckpt'))

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            OmegaConf.save(self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
    
            self.lightning_config.logger_id = pl_module.logger._run_short_id
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

    


class CUDACallback(Callback):
    # modified https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py for lightning 2.0.7
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2 ** 20
        epoch_time = time.time() - self.start_time

        max_memory = trainer.strategy.reduce(max_memory)
        epoch_time = trainer.strategy.reduce(epoch_time)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")

    def root_gpu(self, trainer):
        return trainer.strategy.root_device.index