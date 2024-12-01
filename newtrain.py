import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset
# from S6_model import get_model, save_model
from S6_Model3 import get_model, save_model
from tqdm import tqdm
import os
import argparse

def get_transform(apply_augmentation=False):
    """Get data transformation pipeline with enhanced augmentation"""
    transforms_list = [
        transforms.ToTensor(),  # Convert to tensor first
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    
    if apply_augmentation:
        transforms_list = [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=0,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ] + transforms_list  # Add base transforms after augmentation
        
        # Add RandomErasing after converting to tensor
        transforms_list.append(transforms.RandomErasing(p=0.1))
    
    return transforms.Compose(transforms_list)

def setup_directories():
    """Create necessary directories"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory for saving checkpoints")

def setup_device():
    """Set up and return the device to use"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# Add this at the start of your notebook or training script
def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def load_data_old(use_augmentation, batch_size=128):
    """Load and prepare data loaders"""
    train_transform = get_transform(apply_augmentation=use_augmentation)
    test_transform = get_transform(apply_augmentation=False)
    
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_indices = list(range(50000))
    val_indices = list(range(50000, 60000))
    
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(
        datasets.MNIST('./data', train=True, download=False, transform=test_transform),
        val_indices
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)

# Use it before loading data
set_seed(42)

def load_data(use_augmentation, batch_size=128):
    """Load and prepare data loaders with shuffled split"""
    train_transform = get_transform(apply_augmentation=use_augmentation)
    test_transform = get_transform(apply_augmentation=False)
    
    # Load full training dataset
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    # Generate shuffled indices
    num_train = len(full_train_dataset)
    indices = torch.randperm(num_train)  # Creates a random permutation of indices
    
    # Split indices
    train_size = 50000
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"Split details:")
    print(f"- Total dataset size: {num_train}")
    print(f"- Training set: {len(train_indices)} samples (randomly selected)")
    print(f"- Validation set: {len(val_indices)} samples (randomly selected)")
    
    # Create subsets using shuffled indices
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(
        datasets.MNIST('./data', train=True, download=False, transform=test_transform),
        val_indices
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Additional shuffling during training
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation set
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)

def print_training_config(use_augmentation, initial_lr):
    """Print training configuration details"""
    print("\n=== Training Configuration ===")
    print(f"Initial Learning Rate: {initial_lr}")
    print(f"Optimizer: Adam (betas=(0.9, 0.999), eps=1e-08)")
    print(f"Learning Rate Scheduler: ReduceLROnPlateau")
    print(f" - mode: max (tracking validation accuracy)")
    print(f" - factor: 0.1")
    print(f" - patience: 3 epochs")
    print(f" - min_lr: 1e-6")
    
    print("\n=== Data Augmentation Settings ===")
    if use_augmentation:
        print("Data Augmentation: Enabled for training")
        print(" - Random rotation: ±10 degrees")
        print(" - Random zoom: ±10%")
        print(" - Random shift: ±10% horizontal and vertical")
    else:
        print("Data Augmentation: Disabled")

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler):
    """Train for one epoch"""
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f'Training (lr={scheduler.get_last_lr()[0]:.6f})', leave=False)
    for batch_idx, (data, target) in enumerate(train_pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
        
        train_pbar.set_postfix({
            'loss': f'{train_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return train_loss/len(train_loader), 100 * train_correct / train_total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc='Validation', leave=False)
        for batch_idx, (data, target) in enumerate(val_pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
            
            val_pbar.set_postfix({
                'loss': f'{val_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*val_correct/val_total:.2f}%'
            })
    
    return val_loss/len(val_loader), 100 * val_correct / val_total

def test(model, test_loader, criterion, device):
    """Test the model"""
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for batch_idx, (data, target) in enumerate(test_pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            test_pbar.set_postfix({
                'loss': f'{test_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*test_correct/test_total:.2f}%'
            })
    
    return test_loss/len(test_loader), 100 * test_correct / test_total

def train(use_augmentation=True):
    """Main training function"""
    BATCH_SIZE = 128
    initial_lr = 0.003  # Initial learning rate
    
    print("\n=== Initializing Training Pipeline ===")
    setup_directories()
    device = setup_device()
    
    print("\n=== Preparing Data ===")
    train_loader, val_loader, test_loader, train_size, val_size, test_size = load_data(
        use_augmentation, BATCH_SIZE
    )
    
    print("\n=== Dataset Statistics ===")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {test_size}")
    
    model = get_model().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    
    # Initialize ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Since we're tracking validation accuracy
        factor=0.1,           # Reduce LR by factor of 10
        patience=3,           # Number of epochs with no improvement after which LR will be reduced
        verbose=True,         # Print message when LR is reduced
        min_lr=1e-6,          # Minimum LR
        threshold=0.001,      # Minimum change to qualify as an improvement
        threshold_mode='rel'  # Relative change
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("\n=== Training Configuration ===")
    print(f"Initial Learning Rate: {initial_lr}")
    print(f"Optimizer: Adam (betas=(0.9, 0.999), eps=1e-08)")
    print(f"Scheduler: ReduceLROnPlateau")
    print(f" - mode: max (tracking validation accuracy)")
    print(f" - factor: 0.1")
    print(f" - patience: 3 epochs")
    print(f" - min_lr: 1e-6")
    print(f" - threshold: 0.001")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n=== Starting Training ===")
    best_accuracy = 0.0
    
    for epoch in range(20):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/20 (LR={current_lr:.6f}):")
        
        # Create a simple scheduler wrapper for train_epoch function
        class SimpleScheduler:
            def get_last_lr(self):
                return [current_lr]
        
        temp_scheduler = SimpleScheduler()
        
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device, temp_scheduler)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch Summary:")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}% | "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Step the scheduler with validation accuracy
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_filename = save_model(model, val_accuracy)
            print(f"✓ New best model saved as {model_filename}")
            print(f"Previous best: {best_accuracy:.2f}%")
        
        # Print plateau detection info
        if epoch > scheduler.patience:
            recent_vals = [val_accuracy]  # You might want to keep track of previous accuracies
            max_recent = max(recent_vals)
            print(f"\nPlateau Monitor:")
            print(f"Recent accuracy: {val_accuracy:.2f}% | "
                  f"Best recent: {max_recent:.2f}% | "
                  f"Improvement needed: >{max_recent + scheduler.threshold*max_recent:.2f}%")
        
        if val_accuracy >= 99.4:
            print("\n🎉 Reached target validation accuracy of 99.4%!")
            break
    
    print("\n=== Final Evaluation ===")
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST Model')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                        help='Disable data augmentation')
    parser.set_defaults(augment=True)
    
    args = parser.parse_args()
    train(use_augmentation=args.augment) 