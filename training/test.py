# Torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

# time 
import time
import datetime
import os 

# extras
from typing import Union
from logger import Recorder, WandbRecorder
from config import Config


# tools 
import numpy as np

# Modules
import tools
import dataloader
import losses
import metrics
import transform
from visualize import PointCloudInferenceVisualizer


class Test(object):
    def __init__(self, config: Config, model: nn.Module, recorder: Union[Recorder, None]):
        """
        Initializes the test object with the given config, model, and recorder.
        """
        # Initialize config, recorder, and model
        self.config = config
        self.recorder = recorder
        self.model = model.cuda()  # Move model to GPU

        # Initialize time tracking for remaining time during training
        self.remain_time = tools.RemainTime(self.config.model_config.epochs)

        # Initialize data loader and sampler for testing
        self.test_loader, self.test_sampler = dataloader.load_dataset(config=self.config, recorder=self.recorder)

        # If using distributed training, wrap model with SyncBatchNorm and DDP
        if tools.is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.config.gpu], find_unused_parameters=True)

        # Initialize metrics for evaluation
        self.calib_metrics = metrics.CalibEval(config=self.config, translation_threshold=None, rotation_threshold=None) 
        self.calib_metrics.reset()

        # Average time tracker
        self.avg_time = tools.AverageMeter()
        self.avg_time.reset()

        # Initialize transformation classes for SO3 and SE3
        self.so3 = transform.SO3()
        self.se3 = transform.SE3()

        # Initialize visualization if needed
        if self.config.visualize:
            self.visualizer = PointCloudInferenceVisualizer(os.path.join(self.config.prediction_path, 'images'))

   
    @torch.no_grad()
    def run(self, epoch, print_results=False, save_results_path=None):
        """
        Run the test loop for one epoch, evaluating the model's performance.
        """
        # Set the model to evaluation mode
        self.model.eval()
        model_without_ddp = self.model

        # Reset metrics at the start of evaluation
        self.calib_metrics.reset()

        # If using DataParallel, extract the underlying model
        if hasattr(self.model, 'module'):
            model_without_ddp = self.model.module
        
        # Set epoch for the sampler if present
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)

        # Initialize iteration timer
        total_iter = len(self.test_loader)
        t_start = time.time()

        # Loop through the test data
        for i, data in enumerate(self.test_loader):
            # Model inputs (RGB, depth, point cloud, etc.)
            input_rgb_img = data['img'].cuda()
            input_depth_img = data['uncalibed_depth_img'].cuda() 
            pcd_range = data['pcd_range'].cuda()
            intensity = data['intensity'].cuda()   
            density = data['density'].cuda()   
            input_pcd = data['uncalibed_pcd'].cuda()

            # Ground truth transformations
            gt_intrinsic = data['InTran'][0].cuda()
            gt_extrinsic = data['igt'].cuda()

            # Initial transformation for iterative refinement
            pred_tf = torch.eye(4).repeat(input_rgb_img.size(0), 1, 1).cuda()

            # Time for processing the current batch
            t_process_start = time.time()

            if self.config.model_config.inner_iter == 1:
                # Direct prediction with single iteration
                rot, trans = self.model(input_rgb_img, input_depth_img)
                # Get transformation using Rodrigues formula
                pred_tf = self.se3.exp(torch.cat([rot, trans], dim=1))
                
            else:
                # For iterative refinement with multiple iterations
                depth_generator = transform.DepthImgGenerator(input_rgb_img.shape[-2:], gt_intrinsic, pcd_range, intensity, density, self.config.dataset_config.pooling_size)
            
                # Forward propagation for multiple iterations
                for _ in range(self.config.model_config.inner_iter):
                    # Prediction of rotation and translation
                    rot, trans = self.model(input_rgb_img, input_depth_img)
                    # Get transformation using Rodrigues formula
                    iter_pred = self.se3.exp(torch.cat([rot, trans], dim=1))

                    # Transform the point cloud and re-project to depth image
                    input_depth_img, input_pcd = depth_generator(iter_pred, input_pcd)
                        
                    # Right product (chronologically left product)
                    pred_tf = pred_tf.bmm(iter_pred)

            # Timer for batch processing
            t_process_end = time.time()
            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start
            self.remain_time.update(cost_time=(time.time() - t_start), mode=self.config.mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(epoch=epoch, iters=i, total_iter=total_iter, mode=self.config.mode))
            t_start = time.time()

            # Visualize the result if configured
            if self.config.visualize and i < 100:
                self.visualizer.plot_inference(data, pred_tf, i, False)

            # Update metrics with ground truth and predictions
            self.calib_metrics.add_batch(gt_extrinsic, pred_tf, idx=i)  # 2D predictions

            # Update average time tracker
            if i != 0:
                self.avg_time.update(process_cost_time, self.config.model_config.batch_size_val)

            # Logging the results periodically
            if (i % self.config.log_frequency == 0) or (i == total_iter - 1):
                with torch.no_grad():
                    mean_rot, mean_translation, geodesic = self.calib_metrics.get_stats()
                    
                if self.recorder is not None:
                    # Log string with statistics
                    log_str = '>>> {}  I[{:04d}|{:04d}] DT[{:.4f}] PT[{:.4f}] '.format(
                        self.config.mode, total_iter, i + 1, data_cost_time, process_cost_time)

                    log_str += 'dR [{:0.4f}] dT [{:0.4f}] '.format(np.rad2deg(geodesic[0]), geodesic[1])
                    log_str += ' RT [{}] '.format(remain_time)

                    # Log the string
                    self.recorder.logger.info(log_str)

        # After the loop, get final metrics
        with torch.no_grad():
            rot_err, trans_err, geodesic = self.calib_metrics.get_stats()
            sd_rot, sd_trans = self.calib_metrics.getSD()

            metrics_dict = {
                'roll': rot_err[0],
                'pitch': rot_err[1],
                'yaw': rot_err[2],
                'x': trans_err[0],
                'y': trans_err[1],
                'z': trans_err[2],
                'rot': rot_err.mean().item(),
                'trans': trans_err.mean().item(),
                'dR': geodesic[0],
                'dT': geodesic[1],
                'sd_rot': sd_rot.mean().item(),
                'sd_trans': sd_trans.mean().item(),
                'recall': self.calib_metrics.compute_recall()
            }

        # Log Results at the end of the epoch
        if self.recorder is not None:
            # Log averaged results
            log_str = '>>> {} '.format(self.config.mode)
            log_str += 'AvgT [{:0.4f}] TErr [{:0.4f}] RErr [{:0.4f}] dR [{:0.4f}] dT [{:0.4f}] '.format(
                self.avg_time.avg, trans_err.mean().item(), rot_err.mean().item(), geodesic[0], geodesic[1])     
            self.recorder.logger.info(log_str)

            # Log detailed metrics
            self.recorder.log_message(f"[{self.config.mode}] - " + ', '.join([f'{k}: {v:.4f}' for k, v in metrics_dict.items()]), level='info')

        # Return final result metrics
        result_metrics = {
            'rot': rot_err.mean().item(),
            'trans': trans_err.mean().item(),
        }

        # Save final results
        self.calib_metrics.save_results()

        # If visualization is enabled, generate a video from the images
        if self.config.visualize:
            self.visualize.generate_video_from_images(os.path.join(self.config.root, 'assets', 'output_video.mp4'), 10)

        return result_metrics
