import logging
import os.path as osp
import queue
import sys
import threading
import time
from collections import OrderedDict

import torch
from det3d import torchie

from . import hooks
from .checkpoint import load_checkpoint, save_checkpoint
from .hooks import (CheckpointHook, Hook, IterTimerHook, LrUpdaterHook, OptimizerHook, lr_updater,)
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import (all_gather, get_dist_info, get_host_info, get_time_str, obj_from_dict, synchronize,)

def example_to_device(example, device, non_blocking=False) -> dict:
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "anchors_raw",
                 "anchors_mask_raw", "reg_targets_raw", "reg_weights_raw", "labels_raw"]:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in ["voxels", "bev_map", "coordinates", "num_points", "points", "num_voxels",
                   "voxels_raw", "coordinates_raw", "num_points_raw", "points_raw", "num_voxels_raw",]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch

def parse_second_losses(losses):
    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():

        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        elif loss_name == 'loc_loss_elem_2':
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


class Trainer(object):

    def __init__(self, model, batch_processor, optimizer=None, lr_scheduler=None, work_dir=None, log_level=logging.INFO, logger=None, **kwargs,):
        assert callable(batch_processor)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.batch_processor = batch_processor

        # Create work_dir
        if torchie.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            torchie.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError("'work_dir' must be a str or None")

        # Get model name from the model class
        if hasattr(self.model, "module"):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """
            Examples:
                >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
                >>> type(runner.init_optimizer(optimizer))
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(optimizer, torch.optim, dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be either an Optimizer object or a dict, but got {}".format(type(optimizer)))
        return optimizer

    def _add_file_handler(self, logger, filename=None, mode="w", level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        logging.basicConfig(format="%(asctime)s - %(levelname)s - % (message)s", level=level)
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = "{}.log".format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority="NORMAL"):
        """Register a hook into the hook list.
            Args:
                hook (:obj:`Hook`)
                priority (int or str or :obj:`Priority`)
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # Insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            # self._hooks keep a ascending order;
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        '''Build OptimizerHook/CheckpointHook/IterTimerHook here'''
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):  # True
            assert issubclass(hook_type, Hook)
            return hook_type(**args)  # construction here
        else:
            raise TypeError("'args' must be either a Hook object or dict, not {}".format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)  # self is the param (trainer/runner) of func hook.fn_name

    def load_checkpoint(self, filename, map_location="cpu", strict=False):
        self.logger.info("load checkpoint from %s", filename)
        return load_checkpoint(self.model, filename, map_location, strict, self.logger)

    def save_checkpoint(self, out_dir, filename_tmpl="epoch_{}.pth", save_optimizer=True, meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, "latest.pth")
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # Use relative symlink
        torchie.symlink(filename, linkpath)

    def batch_processor_inline(self, model, data, train_mode, **kwargs):
        '''
            Transfer input data from numpy to torch format;
            Feed data to model and get losses;
        '''

        if "local_rank" in kwargs:
            device = torch.device(kwargs["local_rank"])
        else:
            device = None

        # call_hook here mainly for time calculation
        example = example_to_device(data, torch.cuda.current_device(), non_blocking=False)
        self.call_hook("after_data_to_device")
        if train_mode:
            losses = model(example, return_loss=True)
            self.call_hook("after_forward")
            loss, log_vars = parse_second_losses(losses)
            del losses
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0]))
            self.call_hook("after_parse_loss")
            return outputs
        else:
            return model(example, return_loss=False)

    def train(self, data_loader, epoch, **kwargs):

        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        self._max_iters = self._max_epochs * self.length
        self.call_hook("before_train_epoch") # textLoggerHook: nothing;
        base_step = epoch * self.length

        # for data_batch in BackgroundGenerator(data_loader, max_prefetch=3):
        for i, data_batch in enumerate(data_loader):

            global_step = base_step + i
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(global_step)

            self._inner_iter = i
            self.call_hook("before_train_iter")
            outputs = self.batch_processor_inline(self.model, data_batch, train_mode=True, **kwargs)

            if not isinstance(outputs, dict):
                raise TypeError("batch_processor() must return a dict")

            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])

            self.outputs = outputs
            self.call_hook("after_train_iter")   # optim_hook: backprop;
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def val(self, data_loader, **kwargs):

        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")
        self.logger.info(f"work dir: {self.work_dir}")

        if self.rank == 0:
            prog_bar = torchie.ProgressBar(len(data_loader.dataset))

        detections = {}
        cpu_device = torch.device("cpu")

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook("before_val_iter")
            with torch.no_grad():
                outputs = self.batch_processor(self.model, data_batch, train_mode=False, **kwargs)

            # todo:
            #self.call_hook("after_val_iter")

            for output in outputs:
                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in ["metadata",]:
                        output[k] = v.to(cpu_device)
                detections.update({token: output,})
                if self.rank == 0:
                    for _ in range(self.world_size):
                        prog_bar.update()

        synchronize()
        all_predictions = all_gather(detections)

        if self.rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        # torch.save(predictions, "final_predictions_debug.pkl")
        # TODO fix evaluation module
        result_dict, _ = self.data_loader.dataset.evaluation(predictions, output_dir=self.work_dir)
        self.logger.info("\n")
        for k, v in result_dict["results"].items():
            self.logger.info(f"Evaluation {k}: {v}")

        for k, v in result_dict["results_2"].items():
            self.logger.info(f"Evaluation {k}: {v}")

        self.call_hook("after_val_epoch")

    def resume(self, checkpoint, resume_optimizer=True, map_location="default"):
        if map_location == "default":
            checkpoint = self.load_checkpoint(checkpoint, map_location="cuda:" + str(torch.cuda.current_device()))
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint["meta"]["epoch"]
        self._iter = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """ Start running.
            Args:
                data_loaders (list[:obj:`DataLoader`]);
                workflow (list[tuple]): A list of (phase, epochs) to specify the running order and epochs;
                max_epochs (int);
        """
        assert isinstance(data_loaders, list)
        assert torchie.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info( "Start running, host: %s, work_dir: %s", get_host_info(), work_dir)
        self.logger.info("workflow: %s, max: %d epochs", workflow, max_epochs)
        self.call_hook("before_run")    # for summarywriter

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):

                mode, epochs = flow
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError("Trainer has no method named '{}' to run an epoch".format(mode))
                    epoch_runner = getattr(self, mode)   # todo: get self.train() or self.val()
                elif callable(mode):
                    epoch_runner = mode
                else:
                    raise TypeError("mode in workflow must be a str or callable function not '{}'".format(type(mode)))

                for _ in range(epochs):  # todo: epoches=1 for val mode; epoches=5 for train mode
                    if mode == "train" and self.epoch > max_epochs:
                        return
                    # todo: modified by zhengwu, to eval in last epoch
                    elif mode == "train" and self.epoch == max_epochs:
                        epoch_runner = getattr(self, "val")
                        epoch_runner(data_loaders[1], **kwargs)
                        return
                    elif mode == "val":
                        epoch_runner(data_loaders[i], **kwargs)
                    else:
                        epoch_runner = getattr(self, "train")
                        epoch_runner(data_loaders[i], self.epoch, **kwargs)
                        if 55 <= self.epoch <= 59:
                            epoch_runner = getattr(self, "val")
                            epoch_runner(data_loaders[1], **kwargs)
        self.call_hook("after_run")

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert "policy" in lr_config
            hook_name = lr_config["policy"].title() + "LrUpdaterHook"
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError("'lr_config' must be eigher a LrUpdaterHook object or dict, not '{}'".format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        '''dict(interval=10,hooks=[dict(type="TextLoggerHook"),],)'''
        log_interval = log_config["interval"]
        for info in log_config["hooks"]:
            logger_hook = obj_from_dict(info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority="VERY_LOW")

    def register_training_hooks(self, lr_config, optimizer_config=None, checkpoint_config=None, log_config=None):
        """Register default hooks for training."""
        if optimizer_config is None:  # False
            optimizer_config = {}
        if checkpoint_config is None: # False
            checkpoint_config = {}
        if lr_config is not None:     # False
            assert self.lr_scheduler is None
            self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
