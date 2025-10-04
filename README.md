# AIRL-submission-2025


# Project Submission: ViT on CIFAR-10 and Text-Driven Segmentation with SAM 2

This repository contains two notebooks for the assessment:
1.  `q1.ipynb`: An implementation of a Vision Transformer (ViT) trained on CIFAR-10.
2.  `q2.ipynb`: A demonstration of text-driven image segmentation using GroundingDINO and Segment Anything 2 (SAM 2).

---

## Q1 — Vision Transformer on CIFAR-10

This notebook implements a Vision Transformer from scratch using PyTorch and trains it on the CIFAR-10 dataset.

### How to Run

1.  Open `q1.ipynb` in Google Colab.
2.  Ensure the runtime is set to a GPU (Runtime -> Change runtime type -> T4 GPU).
3.  Run all cells from top to bottom. The training process will take a significant amount of time (approx. 1-2 hours for 100 epochs on a T4 GPU).
4.  The final cell will output the best test accuracy achieved during training.

### Best Model Configuration

The configuration for the best-performing model is defined in a dictionary at the top of the notebook.

```python
config = {
    "dataset": "CIFAR-10",
    "image_size": 32,
    "patch_size": 4,
    "num_classes": 10,
    "embed_dim": 512,
    "depth": 6,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "dropout": 0.1,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 0.05,
    "epochs": 100,
    "warmup_epochs": 10,
    "scheduler": "cosine",
}
```

### Results

The objective was to maximize the overall classification test accuracy.

| Metric                            | Value                                             |
| --------------------------------- | ------------------------------------------------- |
| **Overall Test Accuracy (Best)** | **(Enter the value from your Colab run here)** |
| *Expected Accuracy* | *With this config, >90% is an achievable target.* |


### (Bonus) Analysis

* **Patch Size**: For CIFAR-10's small 32x32 resolution, a patch size of 4x4 was chosen. This creates a sequence of (32/4)^2 = 64 patches. This is a good trade-off; a smaller patch size (e.g., 2x2) would create a very long sequence (256 patches), significantly increasing computational cost, while a larger size (e.g., 8x8) would create a very short sequence (16 patches) and might lose too much local detail.
* **Data Augmentation**: ViTs lack the strong inductive biases of CNNs (like translation equivariance), making them prone to overfitting on smaller datasets. To counter this, strong data augmentation is critical. `TrivialAugmentWide` was used, which is a modern, effective auto-augmentation policy. This, combined with `RandomCrop` and `RandomHorizontalFlip`, was essential for good generalization.
* **Optimizer and Scheduler**: The `AdamW` optimizer with a weight decay of 0.05 was used. This is standard for training transformers as it decouples weight decay from the adaptive learning rate. A `CosineAnnealingLR` scheduler with a 10-epoch linear warmup was crucial. The warmup prevents large, destabilizing updates at the start of training, and the cosine decay helps the model settle into a good minimum.

---

## Q2 — Text-Driven Image Segmentation with SAM 2

This notebook demonstrates a pipeline for segmenting an object in an image using a natural language text prompt.

### Pipeline Description

The process works as follows:
1.  **Input**: The user provides an image URL and a text prompt (e.g., "a red car").
2.  **Text-to-Region (Grounding)**: The image and text prompt are fed into **GroundingDINO**, a zero-shot object detector. It identifies and generates bounding boxes for the object described by the text.
3.  **Region-to-Mask (Segmentation)**: The bounding boxes from GroundingDINO are used as prompts for the **Segment Anything 2 (SAM 2)** model. SAM 2, given the image and the box prompts, produces a high-quality segmentation mask for each object.
4.  **Output**: The final mask is overlaid on the original image for visualization.

### How to Run

1.  Open `q2.ipynb` in Google Colab.
2.  Ensure the runtime is set to a GPU (Runtime -> Change runtime type -> T4 GPU).
3.  Run all cells from top to bottom. The first run will download large model checkpoints for both GroundingDINO and SAM 2, which may take several minutes.
4.  The output will show the original image, the detected bounding boxes, and the final segmentation mask.

### Limitations

* **Dependency on Grounding Model**: The pipeline's success is entirely dependent on the GroundingDINO model's ability to correctly identify the object from the text prompt. If it fails to produce a bounding box, SAM 2 will have no prompt and cannot perform segmentation.
* **Ambiguity**: Ambiguous prompts like "the person" in an image with multiple people may result in either one or all people being detected, depending on the grounding model's behavior. The current implementation segments all detected instances.
* **Computational Cost**: Both models are very large and require a GPU with significant VRAM to run efficiently. The process is not suitable for real-time applications on consumer hardware without significant optimization.
