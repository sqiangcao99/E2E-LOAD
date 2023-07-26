#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""
import pdb

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import src.utils.logging as logging 
import src.utils.metrics as metrics
import src.utils.misc as misc

import src.utils.evalution as evalution

logger = logging.get_logger(__name__)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        cfg,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """
        self._cfg = cfg
        self.iter_timer = Timer() 
        self.data_timer = Timer() 
        self.net_timer = Timer() 
        self.overall_iters = overall_iters # 
        self.multi_label = multi_label # 
        self.ensemble_method = ensemble_method # 
        
        # Initialize tensors.
        # self.video_preds = torch.zeros((num_videos, num_cls)) 
        self.video_preds = {} 
        self.video_labels = {} 

        self.num_clips = {} 
        self.clip_count = {}   
        
        self.stats = {}
        self.results = {}
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        # self.video_preds.zero_()
        self.clip_count.clear()
        self.num_clips.clear() 

        self.video_preds.clear()
        self.video_labels.clear()
        
    def update_stats(self, preds, labels, sessions, frame_indices, num_frames): # 
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            session (tensor): 

        """
        
        preds = preds.softmax(dim=-1).cpu().numpy()

        for bs, (session, frame_indice, num_frame) in enumerate(zip(sessions, frame_indices, num_frames)):
            
            if session not in self.video_preds:
                self.video_preds[session] = np.zeros((num_frame, self._cfg.MODEL.NUM_CLASSES)) 
                
            if session not in self.video_labels:
                self.video_labels[session] = np.zeros((num_frame, self._cfg.MODEL.NUM_CLASSES)) 
                self.num_clips[session] = num_frame.item()
            
            if frame_indice[0] == 0:

                self.video_preds[session][frame_indice] = preds[bs]
                self.video_labels[session][frame_indice] = labels[bs]
                self.clip_count[session] = self._cfg.MODEL.WORK_MEMORY_NUM_SAMPLES

            else:
                self.video_preds[session][frame_indice[-1]] = preds[bs][-1]
                self.video_labels[session][frame_indice[-1]] = labels[bs][-1]
                self.clip_count[session] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter) 
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}/{}".format(cur_iter + 1, self.overall_iters),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(), 
        }
        logging.log_json_stats(stats) 

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self): 
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self,):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        ## probabilities check; 
        check_list = []
        for k, v in self.num_clips.items():
            if v != self.clip_count[k]:
                check_list.append(k)
        
        if len(check_list) > 0:
            logger.warning(
                    '{} get mismatched predictions.'.format(check_list)
                )
            pdb.set_trace() 
        
        self.stats = {"split": "test_final"}

        video_labels = np.concatenate(list(self.video_labels.values()), axis=0)
        video_preds = np.concatenate(list(self.video_preds.values()), axis=0)

        assert len(video_labels) == len(video_preds)

        if self._cfg.DATA.METRICS == "cAP":
            self.results = evalution.eval_perframe(
                self._cfg, video_labels, video_preds
            )

            self.stats["mcAP"] = self.results["mean_AP"]
            
            if self._cfg.TRAIN.PRINT_VERPOSE:
                self.stats["per_class_mcAP"] = self.results["per_class_AP"]

        elif self._cfg.DATA.METRICS == "AP": 
            self.results = evalution.eval_perframe(
                self._cfg, video_labels, video_preds
            )
            self.stats["mAP"] = self.results["mean_AP"]

            if self._cfg.TRAIN.PRINT_VERPOSE:
                self.stats["per_class_mcAP"] = self.results["per_class_AP"]    

        logging.log_json_stats(self.stats)

class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters  
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters  
        self.iter_timer = Timer() 
        self.data_timer = Timer() 
        self.net_timer = Timer() 
        self.loss = ScalarMeter(cfg.LOG_PERIOD)  
        self.loss_total = 0.0
        self.lr = None
        self.grad_norm = None
        # Current minibatch errors (smoothed over a window).

        # Number of misclassified examples.
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.multi_loss = None

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.grad_norm = None

        self.num_samples = 0
        if self.multi_loss is not None:
            self.multi_loss.reset()

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats( 
        self, loss, lr, grad_norm, mb_size, multi_loss=None
    ):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
            multi_loss (list): a list of values for multi-tasking losses.
        """
        self.loss.add_value(loss)  
        self.lr = lr
        self.grad_norm = grad_norm
        self.loss_total += loss * mb_size
        self.num_samples += mb_size 

        if multi_loss: 
            if self.multi_loss is None:
                self.multi_loss = ListMeter(len(multi_loss)) 
            self.multi_loss.add_value(multi_loss) 
        if (
            self._cfg.TRAIN.KILL_LOSS_EXPLOSION_FACTOR
            > 0.0 
            and len(self.loss.deque) > 6
        ):
            prev_loss = 0.0
            for i in range(2, 7):
                prev_loss += self.loss.deque[len(self.loss.deque) - i]
            if (
                loss
                > self._cfg.TRAIN.KILL_LOSS_EXPLOSION_FACTOR * prev_loss / 5.0
            ):
                raise RuntimeError(
                    "ERROR: Got Loss explosion of {} {}".format(
                        loss, datetime.datetime.now()
                    )
                )

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration. # 某个迭代步骤的某个时间步骤;
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        ) 
        eta = str(datetime.timedelta(seconds=int(eta_sec))) 
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "grad_norm": self.grad_norm,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        } 

        if self.multi_loss is not None:
            loss_list = self.multi_loss.get_value()
            for idx, loss in enumerate(loss_list):
                stats["loss_" + str(idx)] = loss  
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "loss": (self.loss_total)
            / (self.num_samples)
            * self._cfg.TRAIN.BATCH_SIZE, 
            "grad_norm": self.grad_norm,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }

        if self.multi_loss is not None:
            avg_loss_list = self.multi_loss.get_global_avg()
            for idx, loss in enumerate(avg_loss_list):
                stats["loss_" + str(idx)] = loss
        logging.log_json_stats(stats, self.output_dir) 

class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.loss = ScalarMeter(cfg.LOG_PERIOD)  #
        self.loss_total = 0.0

        # Min errors (over the full val set).

        # Number of misclassified examples.
        self.num_samples = 0  # ?

        self.all_preds = []  # all the predictions during training
        self.all_labels = []  #

        self.epoch_metric = []  # mAP for THUMOS, mcAP for TVSeries;

        self.output_dir = cfg.OUTPUT_DIR  # 输出路径;

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.data_timer.reset()
        self.net_timer.reset()  # 计时器清零;
        self.num_top1_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        self.loss.reset()
        self.loss_total = 0.0

    def iter_tic(self):
        """
        Start to record time. # 时间开始走表;
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, loss, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            loss (float): .
            mb_size (int): mini batch size.
        """

        self.loss.add_value(loss) 
        self.loss_total += loss * mb_size 
        self.num_samples += mb_size  

    def update_predictions(self, preds, labels):  
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.

        preds = preds.softmax(dim=1).cpu().tolist() 
        # preds = preds.sigmoid().cpu().tolist()
        labels = labels.cpu().tolist()

        self.all_preds.extend(preds)
        self.all_labels.extend(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        
        stats = {
            "_type": "val_iter{}",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(), 
            "eta": eta, 
            "loss": self.loss.get_win_median(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }  

        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "loss": (self.loss_total)
            / (self.num_samples)
            * self._cfg.TRAIN.BATCH_SIZE, 
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            if self._cfg.DATA.METRICS == "cAP":
                results = evalution.eval_perframe(        
                    self._cfg, self.all_labels, self.all_preds
                )

                stats["mcAP"] = results["mean_AP"]
                
                if self._cfg.TRAIN.PRINT_VERPOSE:
                
                    stats["per_class_mcAP"] = results["per_class_AP"]

            elif self._cfg.DATA.METRICS == "AP": 
                
                results = evalution.eval_perframe(
                    self._cfg, self.all_labels, self.all_preds
                )

                stats["mAP"] = results["mean_AP"]

                if self._cfg.TRAIN.PRINT_VERPOSE:
                    stats["per_class_mcAP"] = results["per_class_AP"]

        logging.log_json_stats(stats, self.output_dir)

class EpochTimer:
    """
    A timer which computes the epoch time. 
    """

    def __init__(self) -> None:
        self.timer = Timer()  
        self.timer.reset()  
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer. 
        """
        self.timer.reset()
        self.epoch_times = []  

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset() 

    def epoch_toc(self): 
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)


class ScalarMeter(object): 
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size) 
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_current_value(self):
        return self.deque[-1]

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class ListMeter(object): 
    def __init__(self, list_size):
        """
        Args:
            list_size (int): size of the list.
        """
        self.list = np.zeros(list_size)
        self.total = np.zeros(list_size)
        self.count = 0

    def reset(self):
        """
        Reset the meter.
        """
        self.list = np.zeros_like(self.list)
        self.total = np.zeros_like(self.total)
        self.count = 0

    def add_value(self, value):
        """
        Add a new list value to the meter.
        """
        self.list = np.array(value)
        self.count += 1
        self.total += self.list

    def get_value(self):
        return self.list

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count
