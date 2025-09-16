import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from datasets import build_dataset
from train import train_model
import wandb

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg

def main():
    # Load dataset configuration (blueprint)
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    
    # Initialize W&B run
    run = wandb.init(
        entity="can_g-a",
        project="Wireframe3D",
        config={
            "learning_rate": 0.001,
            "architecture": "PointCloudToWireframe",
            "dataset": "Building3D",
            "epochs": 100,
        },
    )
    
    # Build dataset with preprocessing (blueprint)
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Print dataset info
    print(f"Training samples: {len(building3D_dataset['train'])}")
    print(f"Test samples: {len(building3D_dataset['test'])}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create data loaders for training and testing (blueprint)
    train_loader = DataLoader(
        building3D_dataset['train'], 
        batch_size=3, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=building3D_dataset['train'].collate_batch
    )

    model, loss_history = train_model(train_loader, num_epochs=1000, learning_rate=0.001, wandb_run=run)

    # Save the trained model first
    torch.save(model.state_dict(), 'trained_model.pth')
    
    # Note: All evaluation is handled via evaluate.py
       
    # Save W&B run ID for later use by evaluate.py
    run_id = run.id
    with open('wandb_run_id.txt', 'w') as f:
        f.write(run_id)
    print(f"✓ W&B run ID saved: {run_id}")
    
    # Finish the W&B run
    run.finish()
    
    
if __name__ == "__main__":
    main()
