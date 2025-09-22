"""
Training Pipeline for Crowd Enhancement V2V Model
Handles data loading, training orchestration, and model evaluation
"""

import torch
import torch.utils.data as data
import numpy as np
import cv2
from pathlib import Path
import json
import asyncio
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import yaml
from tqdm import tqdm
import wandb
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .v2v_model import (
    CrowdEnhancementNetwork, 
    CrowdEnhancementTrainer,
    TrainingDataFormat,
    create_model
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrowdDataset(data.Dataset):
    """
    PyTorch Dataset for crowd enhancement training
    Loads multi-pass EXR sequences and corresponding photorealistic targets
    """
    
    def __init__(self, 
                 data_dir: Path,
                 transform=None,
                 sequence_length: int = 1,
                 train_split: bool = True,
                 split_ratio: float = 0.8,
                 cache_data: bool = False):
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        self.cache_data = cache_data
        self.cached_samples = {}
        
        # Load all training samples
        self.samples = self._load_sample_list()
        
        # Split into train/validation
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * split_ratio)
        
        if train_split:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        logger.info(f"Loaded {len(self.samples)} {'training' if train_split else 'validation'} samples")
    
    def _load_sample_list(self) -> List[Dict]:
        """Scan data directory for training samples"""
        samples = []
        
        # Look for .npz training files
        for sample_file in self.data_dir.rglob("*.npz"):
            try:
                # Load metadata to validate sample
                sample_data = np.load(sample_file, allow_pickle=True)
                
                if self._validate_sample(sample_data):
                    samples.append({
                        'file_path': sample_file,
                        'sequence_id': sample_data['metadata'].item()['sequence_id'],
                        'frame_number': sample_data['frame'].item()
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to load sample {sample_file}: {e}")
                continue
        
        return samples
    
    def _validate_sample(self, sample_data) -> bool:
        """Validate that sample contains required data"""
        required_keys = ['frame', 'input', 'metadata']
        
        for key in required_keys:
            if key not in sample_data:
                return False
        
        # Check input passes
        input_data = sample_data['input'].item()
        required_passes = ['beauty', 'depth']
        
        for pass_name in required_passes:
            if pass_name not in input_data or input_data[pass_name] is None:
                return False
        
        return True
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process a training sample"""
        sample_info = self.samples[idx]
        
        # Check cache first
        if self.cache_data and sample_info['file_path'] in self.cached_samples:
            sample_data = self.cached_samples[sample_info['file_path']]
        else:
            sample_data = np.load(sample_info['file_path'], allow_pickle=True)
            if self.cache_data:
                self.cached_samples[sample_info['file_path']] = sample_data
        
        # Extract input passes
        input_dict = sample_data['input'].item()
        
        # Process beauty pass
        beauty_data = input_dict['beauty']['channels']
        beauty_rgb = np.dstack([
            beauty_data.get('R', np.zeros(beauty_data['B'].shape)),
            beauty_data.get('G', np.zeros(beauty_data['B'].shape)),
            beauty_data.get('B', np.zeros(beauty_data['B'].shape))
        ])
        
        # Process depth pass
        depth_data = input_dict['depth']['channels']
        depth_channel = depth_data.get('Z', depth_data.get('depth', np.zeros(beauty_rgb.shape[:2])))
        depth_normalized = self._normalize_depth(depth_channel)
        
        # Process emission pass (shirt colors)
        emission_rgb = np.zeros_like(beauty_rgb)
        if 'emission' in input_dict and input_dict['emission'] is not None:
            emission_data = input_dict['emission']['channels']
            emission_rgb = np.dstack([
                emission_data.get('R', np.zeros(beauty_rgb.shape[:2])),
                emission_data.get('G', np.zeros(beauty_rgb.shape[:2])),
                emission_data.get('B', np.zeros(beauty_rgb.shape[:2]))
            ])
        
        # Process normal pass
        normal_rgb = np.zeros_like(beauty_rgb)
        if 'normal' in input_dict and input_dict['normal'] is not None:
            normal_data = input_dict['normal']['channels']
            normal_rgb = np.dstack([
                normal_data.get('X', np.zeros(beauty_rgb.shape[:2])),
                normal_data.get('Y', np.zeros(beauty_rgb.shape[:2])),
                normal_data.get('Z', np.zeros(beauty_rgb.shape[:2]))
            ])
            # Normalize normals to [-1, 1] range
            normal_rgb = (normal_rgb - 0.5) * 2.0
        
        # Stack all input channels
        input_tensor = np.dstack([
            beauty_rgb,           # 3 channels
            depth_normalized[..., np.newaxis],  # 1 channel
            emission_rgb,         # 3 channels
            normal_rgb            # 3 channels
        ])  # Total: 10 channels
        
        # Create synthetic target (for now - replace with real photorealistic data)
        target_tensor = self._generate_synthetic_target(beauty_rgb, emission_rgb, depth_normalized)
        
        # Apply data augmentation
        if self.transform:
            augmented = self.transform(image=input_tensor, target=target_tensor)
            input_tensor = augmented['image']
            target_tensor = augmented['target']
        
        # Convert to torch tensors
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).float()
        if not isinstance(target_tensor, torch.Tensor):
            target_tensor = torch.from_numpy(target_tensor.transpose(2, 0, 1)).float()
        
        # Normalize to [-1, 1] range for training
        input_tensor = input_tensor * 2.0 - 1.0
        target_tensor = target_tensor * 2.0 - 1.0
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'sequence_id': sample_info['sequence_id'],
            'frame_number': sample_info['frame_number'],
            'metadata': sample_data['metadata'].item()
        }
    
    def _normalize_depth(self, depth_data: np.ndarray) -> np.ndarray:
        """Normalize depth data to [0, 1] range"""
        if depth_data.max() == depth_data.min():
            return np.zeros_like(depth_data)
        
        # Clip extreme values and normalize
        depth_clipped = np.clip(depth_data, np.percentile(depth_data, 1), np.percentile(depth_data, 99))
        depth_normalized = (depth_clipped - depth_clipped.min()) / (depth_clipped.max() - depth_clipped.min())
        
        return depth_normalized.astype(np.float32)
    
    def _generate_synthetic_target(self, beauty: np.ndarray, emission: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Generate synthetic photorealistic target from proxy render
        This is a placeholder - replace with real photorealistic training data
        """
        # Simple enhancement: add noise, adjust colors, enhance clothing areas
        target = beauty.copy()
        
        # Enhance areas with emissive colors (clothing)
        emission_mask = np.any(emission > 0.1, axis=2, keepdims=True)
        
        # Add texture to clothing areas
        if np.any(emission_mask):
            clothing_texture = np.random.normal(0, 0.05, target.shape)
            target = np.where(emission_mask, target + clothing_texture, target)
        
        # Add depth-based atmospheric perspective
        depth_effect = 1.0 - depth[..., np.newaxis] * 0.2
        target = target * depth_effect
        
        # Add subtle color variation
        color_variation = np.random.normal(1.0, 0.02, target.shape)
        target = target * color_variation
        
        # Ensure valid range
        target = np.clip(target, 0, 1)
        
        return target.astype(np.float32)


def get_training_transforms():
    """Data augmentation transforms for training"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(10, 30), p=0.3),
        A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), p=0.5),
    ], additional_targets={'target': 'image'})


def get_validation_transforms():
    """Transforms for validation (no augmentation)"""
    return A.Compose([
        A.Resize(height=512, width=512),
    ], additional_targets={'target': 'image'})


class TrainingConfig:
    """Training configuration dataclass"""
    
    def __init__(self, config_path: Optional[Path] = None):
        # Default configuration
        self.model_config = 'medium'
        self.batch_size = 4
        self.num_epochs = 100
        self.learning_rate = 0.0002
        self.weight_decay = 0.0001
        self.num_workers = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Paths
        self.data_dir = Path("/app/training_data")
        self.output_dir = Path("/app/models/crowd_enhancement")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        
        # Training parameters
        self.save_every_n_epochs = 10
        self.validate_every_n_epochs = 5
        self.early_stopping_patience = 20
        self.gradient_accumulation_steps = 1
        
        # Data augmentation
        self.use_augmentation = True
        self.sequence_length = 1  # For temporal consistency (future feature)
        
        # Wandb logging
        self.use_wandb = False
        self.wandb_project = "crowd-enhancement"
        self.wandb_run_name = None
        
        # Load from config file if provided
        if config_path and config_path.exists():
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: Path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_path: Path):
        """Save configuration to YAML file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        # Convert Path objects to strings for YAML serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def create_data_loaders(config: TrainingConfig) -> Tuple[data.DataLoader, data.DataLoader]:
    """Create training and validation data loaders"""
    
    # Training dataset
    train_transform = get_training_transforms() if config.use_augmentation else None
    train_dataset = CrowdDataset(
        data_dir=config.data_dir,
        transform=train_transform,
        train_split=True,
        cache_data=False  # Set to True if you have enough RAM
    )
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataset
    val_transform = get_validation_transforms()
    val_dataset = CrowdDataset(
        data_dir=config.data_dir,
        transform=val_transform,
        train_split=False,
        cache_data=True  # Cache validation data
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def validate_model(model: CrowdEnhancementNetwork, 
                  val_loader: data.DataLoader, 
                  criterion, 
                  device: str) -> Dict[str, float]:
    """Validate the model on validation set"""
    model.eval()
    
    total_loss = 0.0
    total_l1_loss = 0.0
    total_perceptual_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validating"):
            input_data = batch_data['input'].to(device)
            target_data = batch_data['target'].to(device)
            
            predictions = model(input_data)
            loss_dict = criterion(predictions, target_data)
            
            total_loss += loss_dict['total_loss'].item()
            total_l1_loss += loss_dict['l1_loss'].item()
            total_perceptual_loss += loss_dict['perceptual_loss'].item()
            num_batches += 1
    
    return {
        'val_loss': total_loss / num_batches,
        'val_l1_loss': total_l1_loss / num_batches,
        'val_perceptual_loss': total_perceptual_loss / num_batches
    }


def save_sample_predictions(model: CrowdEnhancementNetwork,
                          val_loader: data.DataLoader,
                          save_dir: Path,
                          device: str,
                          num_samples: int = 8):
    """Save sample predictions for visual inspection"""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if saved_count >= num_samples:
                break
            
            input_data = batch_data['input'].to(device)
            target_data = batch_data['target'].to(device)
            
            predictions = model(input_data)
            pred_images = predictions['enhanced_image']
            
            # Convert tensors to images and save
            for i in range(min(input_data.size(0), num_samples - saved_count)):
                # Convert from [-1, 1] to [0, 1] range
                input_img = (input_data[i][:3] + 1.0) / 2.0  # First 3 channels (beauty)
                target_img = (target_data[i] + 1.0) / 2.0
                pred_img = (pred_images[i] + 1.0) / 2.0
                
                # Convert to numpy and save
                input_np = input_img.cpu().numpy().transpose(1, 2, 0)
                target_np = target_img.cpu().numpy().transpose(1, 2, 0)
                pred_np = pred_img.cpu().numpy().transpose(1, 2, 0)
                
                # Create comparison image
                comparison = np.hstack([input_np, target_np, pred_np])
                comparison = np.clip(comparison * 255, 0, 255).astype(np.uint8)
                
                Image.fromarray(comparison).save(
                    save_dir / f"sample_{saved_count:03d}_input_target_pred.png"
                )
                
                saved_count += 1
                if saved_count >= num_samples:
                    break


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Crowd Enhancement V2V Model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--data-dir", type=str, help="Training data directory")
    parser.add_argument("--output-dir", type=str, help="Output directory for models")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TrainingConfig(Path(args.config) if args.config else None)
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.checkpoint_dir = config.output_dir / "checkpoints"
        config.logs_dir = config.output_dir / "logs"
    if args.wandb:
        config.use_wandb = True
    
    # Create output directories
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save_to_file(config.output_dir / "training_config.yaml")
    
    # Initialize wandb if requested
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__
        )
    
    # Create model
    model = create_model(config.model_config, config.device)
    
    # Create trainer
    trainer = CrowdEnhancementTrainer(
        model=model,
        device=config.device,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and Path(args.resume).exists():
        start_epoch, best_val_loss = trainer.load_checkpoint(Path(args.resume))
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Training loop
    patience_counter = 0
    
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
        
        # Training
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        
        # Validation
        if (epoch + 1) % config.validate_every_n_epochs == 0:
            val_metrics = validate_model(model, val_loader, trainer.criterion, config.device)
            val_loss = val_metrics['val_loss']
            
            logger.info(f"Validation - Loss: {val_loss:.6f}, "
                       f"L1: {val_metrics['val_l1_loss']:.6f}, "
                       f"Perceptual: {val_metrics['val_perceptual_loss']:.6f}")
            
            # Save sample predictions
            if (epoch + 1) % (config.validate_every_n_epochs * 2) == 0:
                sample_dir = config.logs_dir / f"samples_epoch_{epoch + 1:03d}"
                save_sample_predictions(model, val_loader, sample_dir, config.device)
            
            # Log to wandb
            if config.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_l1_loss': val_metrics['val_l1_loss'],
                    'val_perceptual_loss': val_metrics['val_perceptual_loss'],
                    'learning_rate': trainer.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = config.checkpoint_dir / "best_model.pth"
                trainer.save_checkpoint(best_model_path, epoch + 1, best_val_loss)
                logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
        
        # Save regular checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"
            trainer.save_checkpoint(checkpoint_path, epoch + 1, best_val_loss)
    
    logger.info("Training completed!")
    
    # Save final model
    final_model_path = config.checkpoint_dir / "final_model.pth"
    trainer.save_checkpoint(final_model_path, epoch + 1, best_val_loss)
    
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()