# Torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

# time 
import time
import datetime

# tools 
import numpy as np
import wandb

# Modules
import tools
import dataloader
import losses
import metrics
from .warmup_lr import WarmupCosineLR, WarmupLR
import transform

# extras
from typing import Union, Optional, Tuple
from config import Config
from logger import Recorder, WandbRecorder

class Trainer:
    """
    Trainer class to handle model training, validation, and evaluation.

    Args:
        config (Config): Configuration object containing model, training, and dataset settings.
        model (nn.Module): PyTorch model to be trained or validated.
        recorder (Optional[Recorder]): Recorder for logging training details. Can be None.
        wand_recorder (Optional[WandbRecorder]): Wandb recorder for logging metrics and losses. Can be None.
    """
    def __init__(self, config: Config, model: nn.Module, recorder: Optional[Recorder], wand_recorder: Optional[WandbRecorder]):
        # Initialize configuration and logging
        self.config = config
        self.recorder = recorder
        self.wandb_recorder = wand_recorder
        self.model = model.cuda()

        # Initialize training utilities
        self.remain_time = tools.RemainTime(self.config.model_config.epochs)
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = dataloader.load_dataset(config=self.config, recorder=self.recorder)

        # Initialize loss functions and optimizer
        self.criterion = self._initCriterion()
        self.optimizer = self._initOptimizer()

        # Synchronize batch normalization for distributed training
        if tools.is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.config.gpu], find_unused_parameters=False)

        # Initialize metrics for calibration evaluation
        self.calib_metrics = metrics.CalibEval(config=self.config, translation_threshold=None, rotation_threshold=None)
        self.calib_metrics.reset()

        # Transformation utilities
        self.so3 = transform.SO3()
        self.se3 = transform.SE3()

        # Initialize learning rate scheduler
        if self.config.mode == 'train':
            if self.config.model_config.scheduler == 'cosine':
                self.scheduler = WarmupCosineLR(
                    optimizer=self.optimizer,
                    lr=self.config.model_config.lr,
                    warmup_steps=self.config.model_config.warmup_epochs * len(self.train_loader),
                    momentum=self.config.model_config.momentum,
                    max_steps=len(self.train_loader) * (self.config.model_config.epochs - self.config.model_config.warmup_epochs)
                )
            elif self.config.model_config.scheduler == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

        # Mixed precision training
        self.fp16_scaler = torch.cuda.amp.GradScaler() if self.config.model_config.use_fp16 else None

        # Initialize max loss trackers
        self.max_translation_loss = 0
        self.max_rotation_loss = 0
        self.max_photo_loss = 0
        self.max_chamfer_loss = 0

    def _initOptimizer(self) -> torch.optim.Optimizer:
        """
        Initialize the optimizer for the model parameters based on the configuration.

        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        params = self.model.parameters()
        if self.config.model_config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                self.config.model_config.lr,
                momentum=self.config.model_config.momentum,
                weight_decay=self.config.model_config.decay
            )
        elif self.config.model_config.optimizer in ['adam', 'adamw']:
            optimizer = torch.optim.AdamW(
                params=params,
                lr=self.config.model_config.lr,
                weight_decay=self.config.model_config.decay
            )
        else:
            if self.recorder:
                self.recorder.log_message("Invalid Optimizer", level="error")
            raise ValueError(f"Invalid optimizer: {self.config.model_config.optimizer}")
        return optimizer

    def _initCriterion(self) -> dict:
        """
        Initialize loss functions.

        Returns:
            dict: Dictionary containing loss functions.
        """
        criterion = {
            'chamfer_loss': losses.ChamferDistanceLoss(),
            'photo_loss': losses.PhotoLoss(),
            'mse_loss': losses.MSETransformationLoss()
        }
        for _, loss_fn in criterion.items():
            loss_fn.cuda()
        return criterion

    def compute_losses(self, gt_tf, pred_tf, gt_pcd, pred_pcd, gt_depth_img, pred_depth_img) -> Tuple[torch.Tensor, ...]:
        """
        Compute various losses for the model predictions.

        Args:
            gt_tf (torch.Tensor): Ground truth transformation (SE3).
            pred_tf (torch.Tensor): Predicted transformation (SE3).
            gt_pcd (torch.Tensor): Ground truth point cloud.
            pred_pcd (torch.Tensor): Predicted point cloud.
            gt_depth_img (torch.Tensor): Ground truth depth image.
            pred_depth_img (torch.Tensor): Predicted depth image.

        Returns:
            Tuple[torch.Tensor, ...]: Total loss and individual loss components (translation, rotation, photo, chamfer).
        """
        translation_loss, rotation_loss = self.criterion['mse_loss'](pred_tf, gt_tf)
        photo_loss = self.criterion['photo_loss'](gt_depth_img, pred_depth_img)
        chamfer_loss = self.criterion['chamfer_loss'](gt_pcd, pred_pcd)

        # Normalize losses if specified
        if self.config.model_config.normalize_losses:
            translation_loss, rotation_loss, photo_loss, chamfer_loss = self.normalize_losses(
                translation_loss, rotation_loss, photo_loss, chamfer_loss
            )

        # Apply loss weights
        photo_loss *= self.config.model_config.loss_weights_1 * (100.0 if not self.config.model_config.normalize_losses else 1.0)
        chamfer_loss *= self.config.model_config.loss_weights_2
        rotation_loss *= self.config.model_config.loss_weights_3 * (100.0 if not self.config.model_config.normalize_losses else 1.0)
        translation_loss *= self.config.model_config.loss_weights_4 * (100.0 if not self.config.model_config.normalize_losses else 1.0)

        total_loss = photo_loss + chamfer_loss + rotation_loss + translation_loss
        return total_loss, translation_loss, rotation_loss, photo_loss, chamfer_loss

    def normalize_losses(self, translation_loss, rotation_loss, photo_loss, chamfer_loss) -> Tuple[torch.Tensor, ...]:
        """
        Normalize losses based on the maximum observed values during training.

        Args:
            translation_loss (torch.Tensor): Translation loss.
            rotation_loss (torch.Tensor): Rotation loss.
            photo_loss (torch.Tensor): Photo loss.
            chamfer_loss (torch.Tensor): Chamfer loss.

        Returns:
            Tuple[torch.Tensor, ...]: Normalized losses.
        """
        epsilon = 1e-6
        self.max_chamfer_loss = max(self.max_chamfer_loss, chamfer_loss.item())
        self.max_photo_loss = max(self.max_photo_loss, photo_loss.item())
        self.max_rotation_loss = max(self.max_rotation_loss, rotation_loss.item())
        self.max_translation_loss = max(self.max_translation_loss, translation_loss.item())

        return (
            translation_loss / (self.max_translation_loss + epsilon),
            rotation_loss / (self.max_rotation_loss + epsilon),
            photo_loss / (self.max_photo_loss + epsilon),
            chamfer_loss / (self.max_chamfer_loss + epsilon)
        )

    def run(self, epoch, mode='train'):
        """
        Executes the training or validation process for the model.

        Args:
            epoch (int): Current epoch index.
            mode (str): Mode of execution - 'train' or 'val'.

        Returns:
            dict: Resulting metrics including rotation and translation errors.
        """
        # Set dataloader and model state based on the mode
        if mode == 'train':
            dataloader = self.train_loader
            self.model.train()  # Set model to training mode
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
        elif mode == 'val':
            dataloader = self.val_loader
            self.model.eval()  # Set model to evaluation mode
        else:
            raise ValueError('invalid mode: {}'.format(mode))

        # Handle potential distributed training wrapper
        model_without_ddp = self.model
        if hasattr(self.model, 'module'):
            model_without_ddp = self.model.module

        # Initialize metrics and timers
        loss_meter_total = tools.AverageMeter()
        loss_meter_chamfer = tools.AverageMeter()
        loss_meter_photo = tools.AverageMeter()
        loss_meter_rot = tools.AverageMeter()
        loss_meter_trans = tools.AverageMeter()
        self.calib_metrics.reset()  # Reset calibration metrics
        total_iter = len(dataloader)
        t_start = time.time()

        # Iterate through dataloader
        for i, data in enumerate(dataloader):
            t_process_start = time.time()  # Start timer for processing

            # Prepare input data
            input_rgb_img = data['img'].cuda()  # RGB input image
            input_depth_img = data['uncalibed_depth_img'].cuda()  # Uncalibrated depth image

            # Data for loss computation
            pcd_range = data['pcd_range'].cuda()
            intensity = data['intensity'].cuda()
            density = data['density'].cuda()
            gt_depth_img = data['depth_img'].cuda()  # Ground truth depth image

            # Point cloud data
            gt_pcd = data['pcd'].cuda()  # Ground truth point cloud
            input_pcd = data['uncalibed_pcd'].cuda()  # Uncalibrated point cloud

            # Transformation matrices
            gt_intirnsic = data['InTran'][0].cuda()  # Ground truth intrinsics
            gt_extrinsic = data['igt'].cuda()  # Ground truth extrinsics

            # Initialize depth generator for iterative refinement
            depth_generator = transform.DepthImgGenerator(
                input_rgb_img.shape[-2:], gt_intirnsic, pcd_range, intensity, density, self.config.dataset_config.pooling_size
            )
            pred_tf = torch.eye(4, requires_grad=True).repeat(input_rgb_img.size(0), 1, 1).cuda()  # Predicted transformation matrix

            if mode == 'train':
                # Enable mixed precision if applicable
                with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    for _ in range(self.config.model_config.inner_iter):
                        # Forward pass: Predict rotation and translation
                        rot, trans = self.model(input_rgb_img, input_depth_img)

                        # Compute transformation using Rodrigues' formula
                        iter_pred = self.se3.exp(torch.cat([rot, trans], dim=1))

                        # Update depth image and point cloud
                        input_depth_img, input_pcd = depth_generator(iter_pred, input_pcd)

                        # Accumulate predicted transformation
                        pred_tf = pred_tf.bmm(iter_pred)

                    # Compute losses
                    total_loss, translation_loss, rotation_loss, photo_loss, chamfer_loss = self.compute_losses(
                        gt_tf=gt_extrinsic.float(),
                        pred_tf=pred_tf.float(),
                        gt_pcd=gt_pcd,
                        pred_pcd=input_pcd,
                        gt_depth_img=gt_depth_img,
                        pred_depth_img=input_depth_img
                    )

                # Backward propagation and optimizer step
                self.optimizer.zero_grad()
                if self.fp16_scaler is None:
                    total_loss.backward()
                    self.optimizer.step()
                else:
                    self.fp16_scaler.scale(total_loss).backward()
                    self.fp16_scaler.step(self.optimizer)
                    self.fp16_scaler.update()

                # Step learning rate scheduler
                if self.config.model_config.scheduler == "cosine":
                    self.scheduler.step()
            else:
                # Validation loop (no gradient computation)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                        for _ in range(self.config.model_config.inner_iter):
                            # Forward pass: Predict rotation and translation
                            rot, trans = self.model(input_rgb_img, input_depth_img)

                            # Compute transformation using Rodrigues' formula
                            iter_pred = self.se3.exp(torch.cat([rot, trans], dim=1))

                            # Update depth image and point cloud
                            input_depth_img, input_pcd = depth_generator(iter_pred, input_pcd)

                            # Accumulate predicted transformation
                            pred_tf = pred_tf.bmm(iter_pred)

                    # Compute losses for validation
                    total_loss, translation_loss, rotation_loss, photo_loss, chamfer_loss = self.compute_losses(
                        gt_tf=gt_extrinsic.float(),
                        pred_tf=pred_tf.float(),
                        gt_pcd=gt_pcd,
                        pred_pcd=input_pcd,
                        gt_depth_img=gt_depth_img,
                        pred_depth_img=input_depth_img
                    )

            # Update calibration metrics
            self.calib_metrics.add_batch(gt_extrinsic.float(), pred_tf.float())

            # Update loss meters
            loss_meter_chamfer.update(chamfer_loss.item(), self.config.model_config.batch_size if mode == 'train' else self.config.model_config.batch_size_val)
            loss_meter_photo.update(photo_loss.item(), self.config.model_config.batch_size if mode == 'train' else self.config.model_config.batch_size_val)
            loss_meter_rot.update(rotation_loss.item(), self.config.model_config.batch_size if mode == 'train' else self.config.model_config.batch_size_val)
            loss_meter_trans.update(translation_loss.item(), self.config.model_config.batch_size if mode == 'train' else self.config.model_config.batch_size_val)
            loss_meter_total.update(total_loss.item(), self.config.model_config.batch_size if mode == 'train' else self.config.model_config.batch_size_val)

            # Log timing and estimated remaining time
            t_process_end = time.time()
            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start
            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(seconds=self.remain_time.getRemainTime(epoch=epoch, iters=i, total_iter=total_iter, mode=mode))
            t_start = time.time()

            # Timer logger
            t_process_end = time.time()
            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start
            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(epoch=epoch, iters=i, total_iter=total_iter, mode=mode))
            t_start = time.time()

            # Logging
            if (i % self.config.log_frequency == 0) or (i == total_iter-1):
                with torch.no_grad():
                    mean_rot, mean_translation, geodesic = self.calib_metrics.get_stats()
                    
                if self.recorder is not None:
                    for g in self.optimizer.param_groups:
                        lr = g['lr']
                        break

                    # Log string            
                    log_str = '>>> {} E[{:04d}|{:04d}] I[{:04d}|{:04d}] DT[{:.4f}] PT[{:.4f}] '.format(
                        mode, self.config.model_config.epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)

                    log_str += 'LR [{:0.12f}] CL [{:0.8f}] PL [{:0.8f}] RE [{:0.4F}] TE [{:0.4f}] TL [{:0.4f}] MRE [{:0.4f}] MTE [{:0.4f}] '.format(
                        lr, chamfer_loss.item(), photo_loss.item(), rotation_loss.item(), translation_loss.item(), total_loss.item(), mean_rot.mean().item(), mean_translation.mean().item())

                    log_str += ' RT [{}] '.format(remain_time)

                    # Log
                    self.recorder.logger.info(log_str)

        if self.config.model_config.scheduler == "plateau":
            if mode == 'val':
                self.scheduler.step(loss_meter_total.avg)
        
        with torch.no_grad():
            rot_err, trans_err, geodesic = self.calib_metrics.get_stats()
            recall = self.calib_metrics.compute_recall()

            metrics_dict = {
                'roll': rot_err[0] ,
                'pitch': rot_err[1] ,
                'yaw' : rot_err[2] ,
                'x': trans_err[0],
                'y': trans_err[1],
                'z': trans_err[2],
                'rot': rot_err.mean().item(),
                'trans' : trans_err.mean().item(),
                'dR': np.rad2deg(geodesic[0]),
                'dT': geodesic[1]     
            }

            loss_dict = {
                    'chamfer_loss': loss_meter_chamfer.avg,
                    'photo_loss': loss_meter_photo.avg,
                    'rotation_loss': loss_meter_rot.avg,
                    'translation_loss': loss_meter_trans.avg,
                    'total_loss': loss_meter_total.avg
                }
        if self.wandb_recorder is not None and tools.is_main_process():
            self.wandb_recorder.log_losses({**metrics_dict, **loss_dict}, mode, epoch)

        # Log Results 
        if self.recorder is not None:
            # Results at the end of the epoch
            log_str = '>>> {} MCL [{:0.8f}] MPL [{:0.8f}] MRL [{:0.4f}] MTL [{:0.4f}] MTotL [{:0.8f}] '.format(
                mode, loss_meter_chamfer.avg, loss_meter_photo.avg, loss_meter_rot.avg, loss_meter_trans.avg, loss_meter_total.avg)
            
            log_str += 'MTE [{:0.4f}] MRE [{:0.4f}]  Recall [{:0.4f}]  '.format(trans_err.mean().item(), rot_err.mean().item(), recall)     
            self.recorder.logger.info(log_str)

        result_metrics = {
            'rot': rot_err.mean().item(),
            'trans' : trans_err.mean().item(),
        }

        return result_metrics