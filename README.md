# Fashion MNIST Classifier

A multiclass image classification project built with PyTorch to classify images from the Fashion-MNIST dataset.

## Dataset

The project uses the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 pixels
- 10 clothing and footwear classes

## Model

A fully connected neural network (MLP) is used for classification.

```text
28 × 28 Image
     ↓
   Flatten
     ↓
 784 → 64
     ↓
   ReLU
     ↓
  64 → 32
     ↓
   ReLU
     ↓
  32 → 10
     ↓
   Logits
```

The output layer does not use an activation function, as the raw logits are passed directly to `CrossEntropyLoss`.

## Training
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 20
- Activation: ReLU

## Evaluation
The model is evaluated using Loss, Accuracy, Precision, Recall and F1 Score

## Results
| Metric | Training | Test |
|---|---:|---:|
| Loss | 0.2085 | 0.3531 |
| Accuracy | 92.13% | 88.19% |
| Precision | 92.10% | 88.37% |
| Recall | 92.13% | 88.19% |
| F1 Score | 92.10% | 88.20% |

## Technologies
- Python
- PyTorch
- Torchvision
- TorchMetrics
- Matplotlib
- Jupyter Notebook