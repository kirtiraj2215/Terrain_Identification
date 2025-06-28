# Terrain Identification using Semantic Segmentation

This project performs semantic segmentation of terrain types (like grass, rocks, trail, water, mud) using DeepLabV3+ with a ResNet34 encoder, trained on custom annotated terrain datasets.

## Project Overview

The goal is to enable machines (e.g., ground robots, drones) to **visually identify and classify terrain types** from RGB images for downstream applications like terrain-aware navigation or path planning.

## Repository Structure

| File/Folder | Purpose |
|-------------|---------|
| `TerrainIdentification.ipynb` | The full Colab notebook used for training, evaluation, and visualization |
| `best_model.pth` (Link - https://drive.google.com/drive/folders/1crH84WOqywiG5UcFF_k-d04oc8QEWwBK?usp=sharing) | Trained DeepLabV3+ model weights (PyTorch `.pth`) |
| `README.md` | Project overview and instructions |

## Model Details

- **Architecture**: DeepLabV3+
- **Backbone**: ResNet34 (ImageNet pretrained)
- **Input size**: 512 × 512 RGB images
- **Classes**:
  - 0: Background
  - 1: Grass
  - 2: Water
  - 3: Rocks
  - 4: Mud / Trail

## Requirements

This project was run in **Google Colab** with the following libraries:

```bash
torch
torchvision
segmentation_models_pytorch
albumentations
opencv-python
matplotlib
PIL
```

## Training Pipeline Summary

1. Dataset loaded from structured `raw/` and `annotations/` folders
2. Custom PyTorch Dataset class handles images and label conversion
3. DeepLabV3+ model trained using CrossEntropyLoss
4. Checkpoints saved when validation loss improves

## Inference Example

You can use the trained model to predict terrain classes from new images. Example visualization:
- **Left**: Input RGB image
- **Right**: Predicted terrain mask (color-coded)

To test locally or on new data, load the model and run predictions as shown in the notebook.

## How to Use

### 1. Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/terrain-identification.git
cd terrain-identification
```

**Or open the notebook directly in Google Colab for GPU-based training**

### 2. Run the notebook:

Open `TerrainIdentification.ipynb` and run each cell step by step.

### 3. Use the trained model:

You can download or re-use `best_model.pth` to run inference on any 512×512 image.

## Author

**Kirtiraj Tilakdhari Jamnotiya**  
📧 kjamnotiya@gmail.com
