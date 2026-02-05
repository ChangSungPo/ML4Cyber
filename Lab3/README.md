# Lab 3: Breaking CAPTCHAs with Convolutional Neural Networks

Build and train a deep learning model to automatically recognize CAPTCHA text using PyTorch.

## What You'll Learn

| Topic | Description |
|-------|-------------|
| **Preprocessing** | Load CAPTCHA images, extract individual characters using contour detection |
| **Feature Extraction** | Convert character images to normalized tensors, encode labels |
| **Neural Network Design** | Build a CNN with convolutional layers, pooling, and fully-connected layers |
| **Training Loop** | Implement forward pass, loss computation, backpropagation, and gradient updates |
| **Evaluation** | Test end-to-end CAPTCHA recognition pipeline and measure accuracy |

## Quick Start

```bash
# Install dependencies
pip install matplotlib scikit-learn "opencv-python>4" imutils tqdm torch

# Verify required files exist:
# - lab_3_helpers.py (helper functions)
# - captcha-images/ (dataset directory)

# Open notebook
jupyter notebook lab-3-student-20250121.ipynb
```

## The 3 Exercises

| # | Task | Key Concepts |
|:-:|------|--------------|
| 1 | Build CNN model architecture | Conv2d, ReLU, MaxPool2d, Flatten, Linear |
| 2 | Implement training loop | Forward pass, loss, backward pass, optimizer step |
| 3 | Run evaluation pipeline | Model inference, predictions, accuracy computation |

## ML Pipeline Overview

```
CAPTCHA Images → Extract Characters → Normalize Features → Train CNN → Evaluate
      ↓                ↓                    ↓               ↓           ↓
  captcha-images   contour detection   20x20 tensors   Training loop  Accuracy
```

## Key Concepts

### Contour Detection
Uses OpenCV to find bounding boxes around characters within CAPTCHA images:
- Binary thresholding to isolate text
- Contour detection to identify character regions
- Width-to-height ratio to split conjoined characters
- Sorting by X-coordinate to preserve character order

### Convolutional Neural Network (CNN)
A deep learning architecture for image classification:
- **Convolutional layers**: Extract local features (edges, textures)
- **Activation (ReLU)**: Introduce non-linearity
- **Pooling layers**: Reduce spatial dimensions, prevent overfitting
- **Fully-connected layers**: Learn decision boundaries
- **Cross-entropy loss**: Measure classification error

### PyTorch Training Loop
Explicit training process with full control:
1. **Forward pass**: Input → Model → Predictions (logits)
2. **Loss computation**: Compare predictions vs. ground truth
3. **Backward pass**: Compute gradients via backpropagation
4. **Optimizer step**: Update weights in direction of negative gradient
5. **Gradient clearing**: Reset gradients for next iteration

## PyTorch APIs Reference

Key PyTorch modules and functions you'll use in this lab:

| API | Purpose |
|-----|---------|
| `torch.as_tensor()` | Convert NumPy array to PyTorch tensor |
| `torch.nn.Sequential()` | Container for stacking layers into a model |
| `torch.nn.Conv2d()` | 2D convolutional layer for feature extraction |
| `torch.nn.ReLU()` | Rectified Linear Unit activation function |
| `torch.nn.MaxPool2d()` | Max pooling layer to reduce spatial dimensions |
| `torch.nn.Flatten()` | Reshape multi-dimensional tensor to 1D |
| `torch.nn.Linear()` | Fully-connected (dense) layer |
| `torch.nn.functional.cross_entropy()` | Combined softmax + negative log-likelihood loss |
| `torch.optim.AdamW()` | Adaptive learning rate optimizer |
| `model.forward()` or `model()` | Pass input through model to get predictions |
| `loss.backward()` | Compute gradients via backpropagation |
| `optimizer.step()` | Update model parameters using computed gradients |
| `optimizer.zero_grad()` | Clear accumulated gradients from previous batch |
| `model.train()` | Set model to training mode (enables dropout, batch norm updates) |
| `model.eval()` | Set model to evaluation mode (freezes batch norm, disables dropout) |
| `tensor.argmax()` | Find index of maximum value (predict class) |
| `tensor.numpy()` | Convert PyTorch tensor to NumPy array |
| `torch.save()` | Save model weights to file |
| `torch.load()` | Load model weights from file |

**Example Usage:**
```python
import torch
from torch import nn, optim
from torch.nn import functional as f

# Create tensor from NumPy array
features = torch.as_tensor(features_np, dtype=torch.float32)

# Build model
model = nn.Sequential(
    nn.Conv2d(1, 20, (5, 5), padding="same"),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(1250, 500),
    nn.ReLU(),
    nn.Linear(500, n_classes)
)

# Training loop
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
model.train()

for X_batch, y_batch in dataloader:
    logits = model(X_batch)                          # Forward pass
    loss = f.cross_entropy(logits, y_batch)         # Compute loss
    loss.backward()                                  # Backward pass
    optimizer.step()                                 # Update weights
    optimizer.zero_grad()                            # Clear gradients

# Evaluation
model.eval()
predictions = model(features).argmax(dim=-1).numpy()
torch.save(model.state_dict(), "weights.pt")       # Save weights
```

## File Structure

| File | Description |
|------|-------------|
| `lab-3-student-20250121.ipynb` | Your working notebook with TODOs |
| `lab_3_helpers.py` | Helper functions for image processing |
| `captcha-images/` | Dataset of CAPTCHA images (filename = ground truth label) |
| `char-images-31528476/` | Extracted individual character images (created during preprocessing) |
| `labels.pkl` | Saved label encoder for character-to-index mapping |
| `captcha-model-weights.pt` | Trained model weights (created after training) |

## Expected Results

- **Model accuracy on test set**: ~95-98%
- **Training time**: ~2-5 minutes (CPU), ~30-60 seconds (GPU)
- **Total lab time**: ~90-120 minutes

## Common Issues

| Problem | Solution |
|---------|----------|
| Missing helper file | Ensure `lab_3_helpers.py` is in same directory |
| Missing dataset | Download/extract `captcha-images/` |
| `ModuleNotFoundError: torch` | Run: `pip install torch` |
| Out of memory | Reduce `BATCH_SIZE` in training cell |
| GPU not found | Model will fall back to CPU automatically |
| Character extraction failures | ~5-10% of CAPTCHAs may fail; these are marked with "-" |

## Tips for Success

1. **Understand each step**: Read cell comments carefully before implementing TODOs
2. **Test incrementally**: Run cells sequentially; don't skip ahead
3. **Monitor training**: Check loss values decrease each epoch
4. **Inspect failures**: Use visualization cells to understand extraction failures
5. **Experiment**: Try different batch sizes, epochs, or learning rates

## References

1. How to break a CAPTCHA system in 15 minutes with Machine Learning: https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
2. CaptchaSolver Jupyter Notebook: https://github.com/BenjaminWegener/CaptchaSolver
3. PyTorch Official Website and Installation: https://pytorch.org
4. PyTorch Documentation: https://docs.pytorch.org/docs/stable/index.html
5. PyTorch CNN Tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
