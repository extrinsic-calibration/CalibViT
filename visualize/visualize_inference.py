import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from typing import Optional
import copy

from dataloader.dataset_utils import PointCloudProjection

class PointCloudInferenceVisualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def transform_point_cloud(point_cloud, transformation_matrix):
        point_cloud = np.array(point_cloud)[0].T
        transformation_matrix = np.array(transformation_matrix)

        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 matrix.")
        if point_cloud.shape[1] != 3:
            raise ValueError("Point cloud must have 3 columns (x, y, z).")

        homogenous_point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        transformed_point_cloud = np.dot(homogenous_point_cloud, transformation_matrix.T)
        return transformed_point_cloud[:, :3]

    @staticmethod
    def project_points(point_cloud, intrinsic_matrix):
        projected_points = np.dot(intrinsic_matrix, point_cloud.T).T
        projected_points /= projected_points[:, 2].reshape(-1, 1)
        return projected_points

    @staticmethod
    def plot_image_with_points(image, points, coloring, title, output_path, show=False):
        plt.figure(figsize=(12, 6), dpi=300)
        plt.imshow(image)
        plt.scatter(points[:, 0], points[:, 1], c=coloring, s=10)
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        if show:
            #plt.title(title)
            plt.show()
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',pad_inches=-0.1)
            plt.close()

    def plot_inference(self, data, predicted_tf, idx, all_points=True, show=False):
        image = data['img'].cpu().numpy()[0].transpose(1, 2, 0)
        point_cloud_uncalib = data['uncalibed_pcd'].detach().numpy()
        point_cloud_calib = data['pcd'].detach().numpy()[0].T
        intrinsic_matrix = data['InTran'].detach().numpy()[0]
        predicted_tf = predicted_tf.cpu().detach().numpy()[0]

        projector = PointCloudProjection()
        point_cloud_pred = self.transform_point_cloud(copy.copy(point_cloud_uncalib), predicted_tf)
        point_cloud_uncalib = point_cloud_uncalib[0].T

        if not all_points:
            # Uncalibed
            u, v, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_uncalib.T)
            point_cloud_uncalib = point_cloud_uncalib[r]

            # Predicted 
            u, v, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_pred.T)
            point_cloud_pred = point_cloud_pred[r]

            # Ground Truth 
            u, v, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_calib.T)
            point_cloud_calib = point_cloud_calib[r]

        coloring_uncalib = point_cloud_uncalib[:, 2]
        coloring_pred = point_cloud_pred[:, 2]
        coloring_calib = point_cloud_calib[:, 2]

        projected_points_uncalib = self.project_points(point_cloud_uncalib, intrinsic_matrix)
        projected_points_pred = self.project_points(point_cloud_pred, intrinsic_matrix)
        projected_points_calib = self.project_points(point_cloud_calib, intrinsic_matrix)

        output_path_uncalibed = os.path.join(self.output_dir, f"plot_uncalibed_{idx}.png")
        output_path_pred = os.path.join(self.output_dir, f"plot_pred_{idx}.png")
        output_path_gt = os.path.join(self.output_dir, f"plot_gt_{idx}.png")

        self.plot_image_with_points(image, projected_points_uncalib, coloring_uncalib, 'Projected Points on Image Input', output_path_uncalibed, show)
        self.plot_image_with_points(image, projected_points_pred, coloring_pred, 'Projected Points on Image Predicted', output_path_pred, show)
        self.plot_image_with_points(image, projected_points_calib, coloring_calib, 'Projected Points on Image GT', output_path_gt, show)

    
    def generate_video_from_images(
        self, 
        output_video: str, 
        frame_rate: int = 30
    ) -> None:
        """
        Generates a video by stacking input and predicted images side-by-side and saving it to a file.

        Parameters:
            output_video (str): Path to the output video file (e.g., 'output.mp4').
            frame_rate (int): Frames per second for the output video (default is 30).

        Returns:
            None: This function does not return any value. It generates and saves the video.
        
        Raises:
            ValueError: If the number of input and prediction images do not match.
            FileNotFoundError: If any input or prediction image cannot be read.
        """
        # Get all input and prediction images
        inputs = sorted([f for f in os.listdir(self.output_dir) if f.startswith("plot_uncalibed")])
        inputs = sorted(inputs, key=lambda x: int(''.join(filter(str.isdigit, x))))[0:100]

        prediction = sorted([f for f in os.listdir(self.output_dir) if f.startswith("plot_pred")])
        prediction = sorted(prediction, key=lambda x: int(''.join(filter(str.isdigit, x))))[0:100]

        # Check if the number of input and prediction images match
        if len(inputs) != len(prediction):
            raise ValueError("The number of input and prediction images must be the same.")
        
        # Read the first pair of images to determine size
        input_img = cv2.imread(os.path.join(self.output_dir, inputs[0]))
        pred_img = cv2.imread(os.path.join(self.output_dir, prediction[0]))

        if input_img is None or pred_img is None:
            raise FileNotFoundError("Could not read input or prediction images.")

        height_in, width_in, _ = input_img.shape
        height_out, width_out, _ = pred_img.shape

        # Initialize the video writer with combined width and height
        combined_width = width_in + width_out  # Combined width for the horizontal stack
        video_writer = cv2.VideoWriter(
            output_video, 
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 codec
            frame_rate, 
            (combined_width, height_in)
        )

        # Text for labeling the images
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (20, 50)
        fontScale = 2
        color = (0, 0, 255)
        thickness = 10

        # Process each input and prediction image
        for input_file, pred_file in zip(inputs, prediction):
            # Read input and prediction images
            input_img = cv2.imread(os.path.join(self.output_dir, input_file))
            pred_img = cv2.imread(os.path.join(self.output_dir, pred_file))

            if input_img is None or pred_img is None:
                raise FileNotFoundError(f"Could not read {input_file} or {pred_file}")

            # Resize images to fit in the final video frame
            input_img = cv2.resize(input_img, (1024, 512))
            pred_img = cv2.resize(pred_img, (1024, 512))

            # Add text labels to the images
            input_img = cv2.putText(input_img, 'Decalibrated Input', org, font, fontScale, color, thickness, cv2.LINE_AA)
            pred_img = cv2.putText(pred_img, 'Calibrated Output', org, font, fontScale, color, thickness, cv2.LINE_AA)

            # Stack the images horizontally
            combined_img = np.hstack([input_img, pred_img])

            # Write the combined image to the video
            video_writer.write(combined_img)
        
        # Release resources
        video_writer.release()
        print(f"Video saved to {output_video}")