## 📖 Abstract  
Sensor calibration for autonomous vehicles is commonly performed offline. However, continuous vibrations during operation lead to de-calibration, necessitating periodic correction. We present an online calibration network, CalibViT, which estimates the extrinsic calibration matrix between the camera and LiDAR with high accuracy and robustness. Leveraging transformer models, known for their ability to capture long-range dependencies and global context, CalibViT is particularly effective in calibration tasks where understanding complex spatial alignments between sensors is essential. We introduce two variants of our approach: CalibViT-v1, which employs a CNN backbone followed by a cross-attention block, and CalibViT-v2, which utilizes a Transformer backbone leveraging self-attention, followed by the cross-attention block. We evaluate our networks on NuScenes, KITTI Odometry, and TruckScenes datasets, demonstrating the effectiveness of our method across diverse real-world scenarios.


### 📊 Calibration with SotA

| Model                            | Type          | Rx    | Ry    | Rz    | Rm    | Tx    | Ty    | Tz    | Tm    | Runtime [ms] | Range                          |
|----------------------------------|---------------|------------|------------|------------|------------|------------|------------|------------|------------|--------------|--------------------------------|
| **[CalibNet](https://arxiv.org/abs/1803.08181)**          | Iterative     | 0.1500     | 0.9000     | 0.1800     | **0.4100** | 4.200      | 1.600      | 7.220      | **4.340**  | 44           | ±10°, ±0.20m                     |
| **[CalibRCNN](https://ieeexplore.ieee.org/document/9341147)**         | Multi-frame   | 0.2100     | 2.2100     | 0.5000     | **0.9733** | 7.800      | 3.200      | 6.200      | **5.733**  | -            | ±10°, ±0.25m                     |
| **[CalNet](https://ieeexplore.ieee.org/document/9341147)**               | One-shot      | 0.1000     | 0.3800     | 0.1200     | **0.2000** | 3.650      | 1.630      | 3.800      | **3.030**  | **21**       | ±10°, ±0.25m                     |
| **[CalibDNN](https://arxiv.org/abs/2103.14793)**           | Iterative     | 0.1100     | 0.3500     | 0.1800     | **0.2100** | 3.800      | 1.800      | 9.600      | **5.070**  | -            | ±10°, ±0.25m                     |
|**[PSNet](https://ieeexplore.ieee.org/document/9810306)**                | One-shot     | _0.0600_   | **0.2600** | _0.1200_   | **_0.1500_**| 3.800      | 2.800      | 2.600      | **3.100**  | -            | ±10°, ±0.25m                     |
| **CalibViT-V1**        | One-shot      | 0.0841     | 0.3319     | 0.1266     | **0.1809** | _2.815_    | **0.983**  | _2.497_    | **2.0986** | 50           | ±10°, ±0.25m                     |
| **CalibViT-V2**        | One-shot      | **0.0578** | _0.2698_   | **0.0890** | **0.1389** | **2.247**  | _1.019_    | **2.323**  | **1.863**  | _35_         | ±10°, ±0.25m                     |
___


| Model                            | Type          | Rx    | Ry    | Rz    | Rm    | Tx    | Ty    | Tz    | Tm    | Runtime [ms] | Range                          |
|----------------------------------|---------------|------------|------------|------------|------------|------------|------------|------------|------------|--------------|--------------------------------|
| **[RegNet](https://arxiv.org/abs/1707.03167)**       | Cascaded      | 0.2400     | 0.2500     | 0.3600     | **0.2833** | 7.000      | 7.000      | 4.000      | **6.000**  | 94           | ±20°, ±1.50m                     |
| **[LCCNet](https://arxiv.org/abs/2012.13901)**              | Cascaded      | 0.3090     | 0.3300     | 0.3340     | **0.3200** | **1.267**  | 2.212      | **1.107**  | **1.528**  | **18**       | ±5°, ±0.50m                      |
| **CalibViT-V1**        | One-shot      | _0.0971_   | 0.3439     | _0.1599_   | **0.2003** | 3.662      | _1.244_    | 2.5219     | **2.4762** | 50           | ±5°, ±0.50m                      |
| **CalibViT-V2**        | One-shot      | **0.0540** | **0.2500** | **0.0582** | **0.1207** | _2.476_    | **0.938**  | _2.019_    | **1.8124** | _35_         | ±5°, ±0.50m                      |
___


<div align="center">
  <video src="https://github.com/user-attachments/assets/c07b804b-0bc1-4025-b5e6-11779400bf26" />
</div>
      
___
## 🚀 Environment Setup

To run the project efficiently, we use Docker to ensure consistency and reproducibility. Follow these steps to set up and run the environment:

---

### Build the Docker Image

Make sure you have Docker installed on your system. Then, build the Docker image using the provided `Dockerfile`:

```bash
docker build -t calibvit:latest -f Dockerfile .
docker run  --name calibenv \
    --privileged --gpus device='all,"capabilities=compute,utility,graphics"' \
    -it --network host --shm-size 64GB \
    -v $(pwd)/CalibViT:/workspace \
    -v $(pwd)/datasets:/workspace/datasets calibvit:latest /bin/bash

cd /workspace
```
___
### 📦 Folder Structure

The project is organized as follows:

```bash
workspace/
├── Dockerfile
├── LICENSE
├── README.md
├── assets/
├── config/
│   ├── model_config.json
├── dataloader/
├── datasets/
│   ├── kitti/
│   ├── nuscenes/
├── index.html
├── logger/
├── losses/
├── main.py
├── metrics/
├── models/
│   ├── __init__.py
│   ├── calibnet/
│   ├── calibvit/
│   ├── model_loader.py
├── notebooks/
├── results/
├── tools/
├── training/
├── transform/
├── visualize/
├── weights/
│   ├── checkpoint/
│   ├── pretrained/
│   ├── test_decalib/
│   ├── validation_decalib/
```
___
### 🎯 Training and Validation 
```bash
torchrun --nproc_per_node=1 --master_port=63545  main.py --version=v2 \
         --mode=train --batch_size=32 --max_deg=10 --max_tran=0.25 \
         --distribution=uniform --dataset=kitti --epoch=100 --count=1 

torchrun --nproc_per_node=1 --master_port=63545  main.py --version=v2 --mode=test \
         --batch_size=1 --max_deg=10 --max_tran=0.25 --distribution=uniform --dataset=kitti --epoch=100 --count=1 

```
___
### 🛠️ Integration: Adding Your Custom Model 

This framework allows you to easily add and integrate custom models. Follow these steps:

1. Add Model Implementation: Create a folder for your model inside ```models/```. For example, if your model is called MyModel, you should add it to ```models/mymodel/```.

2. Update Configuration: Add your model details to ```config/model_config.json```. For example:
```
{
    "myModel": {
        "dataset" : {
            "model": "myModel",
            "param1" :  "value1" 
            "param2" :  value2
        }
    }    
}
```
3. Load Your Model: Update ```models/model_loader.py``` to include your model class and import logic.

___

### 🙌 Acknowledgments

We would like to acknowledge the invaluable contributions of open-source repositories and research papers that supported this work:

##### GitRepos
1. **[Swin Transformer Repository](https://github.com/microsoft/Swin-Transformer)**  
2. **[CalibNet](https://github.com/gitouni/CalibNet_pytorch)**  
3. **[RangeViT](https://github.com/valeoai/rangevit)**  

##### Research Papers
1. **[CalibNet: Geometrically Supervised Extrinsic Calibration using 3D Spatial Transformer Networks](https://arxiv.org/abs/1803.08181)**

2. **[PSNet: LiDAR and Camera Registration Using Parallel Subnetworks](https://ieeexplore.ieee.org/document/9810306)**

3. **[LCCNet: LiDAR and Camera Self-Calibration using Cost Volume Network](https://arxiv.org/abs/2012.13901)**

4. **[CalibRCNN: Calibrating Camera and LiDAR by Recurrent Convolutional Neural Network and Geometric Constraints](https://ieeexplore.ieee.org/document/9341147)**

5. **[CalibDNN: Multimodal Sensor Calibration for Perception Using Deep Neural Networks](https://arxiv.org/abs/2103.14793)**

5. **[RegNet: Multimodal Sensor Registration Using Deep Neural Networks](https://arxiv.org/abs/1707.03167)**

5. **[CALNet: LiDAR-Camera Online Calibration With Channel Attention and Liquid Time-Constant Network](https://ieeexplore.ieee.org/document/9956145)**

We are deeply grateful for the open-source and academic communities' efforts to share resources, enabling the advancement of research in the domain of vision-based calibration.
___
