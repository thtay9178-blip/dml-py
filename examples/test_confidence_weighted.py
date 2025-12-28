"""
Quick test for Confidence-Weighted DML (1 epoch only)
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pydml.models.cifar import resnet32, wrn_28_10
from pydml.trainers.confidence_weighted import ConfidenceWeightedDML, ConfidenceWeightedConfig

# Minimal setup for quick testing
print("=" * 70)
print("QUICK CONFIDENCE-WEIGHTED DML TEST")
print("=" * 70)

# Use tiny subset of data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Use only 200 samples for quick test
trainset = Subset(trainset, list(range(200)))
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testset = Subset(testset, list(range(100)))
testloader = DataLoader(testset, batch_size=32)

# Create two small models
models = [
    resnet32(num_classes=10),
    wrn_28_10(num_classes=10)
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# Configure Confidence-Weighted DML
config = ConfidenceWeightedConfig(
    temperature=3.0,
    confidence_threshold=0.6,
    weighting_strategy='softmax',
    min_weight=0.1,
    max_weight=2.0
)

# Initialize trainer
trainer = ConfidenceWeightedDML(
    models=models,
    config=config,
    learning_rate=0.01,
    device=device
)

print("Running 1 epoch for testing...")
print("-" * 70)

# Train for just 1 epoch
history = trainer.fit(trainloader, testloader, epochs=1)

print("\n" + "=" * 70)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val accuracy: {history['val_acc'][-1]:.2f}%")

# Print confidence statistics
if 'confidence_stats' in history:
    stats = history['confidence_stats'][-1]
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {stats['mean_confidence']:.4f}")
    print(f"  Std:  {stats['std_confidence']:.4f}")
    print(f"  Above threshold: {stats['above_threshold_ratio']:.2%}")
