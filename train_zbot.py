"""
ZBot Robust Walking on Rough Terrain
===================================

This script implements robust walking for the ZBot humanoid robot across different terrains.
The approach follows these steps:
1. Train a robust walking policy on flat terrain with domain randomization
2. Adapt the policy to rough terrain
3. Evaluate and visualize the results

Author: Claude
Date: 2024
"""

#######################
# Setup & Dependencies
#######################

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import jax
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import config_dict
from playground.zbot import joystick as zbot_joystick
from playground.zbot import randomize as zbot_randomize
from playground.zbot import zbot_constants
from playground.runner import ZBotRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('zbot_training.log')
    ]
)
logger = logging.getLogger(__name__)

########################
# Training Configuration 
########################

def create_training_args(task="flat_terrain", load_existing=False):
    """Create training arguments with enhanced settings"""
    args = argparse.Namespace(
        env="ZbotJoystickFlatTerrain",
        task=task,
        debug=False,
        save_model=True,
        load_model=load_existing,
        seed=42,
        num_episodes=100,
        episode_length=5000,
        x_vel=1.0,
        y_vel=0.0,
        yaw_vel=0.0,
        curriculum=True,
        initial_roughness=0.0,
        final_roughness=1.0,
        roughness_increment=0.1,
        min_success_rate=0.7,
        learning_rate=3e-4,
        batch_size=256,
        update_epochs=10,
        randomize_mass=True,
        mass_range=(0.8, 1.2),
        randomize_friction=True,
        friction_range=(0.8, 1.2)
    )
    return args

def plot_training_progress(runner, title):
    """Plot training progress with error bands"""
    plt.figure(figsize=(10, 6))
    plt.plot(runner.x_data, runner.y_data, label='Mean Reward')
    plt.fill_between(
        runner.x_data,
        np.array(runner.y_data) - np.array(runner.y_dataerr),
        np.array(runner.y_data) + np.array(runner.y_dataerr),
        alpha=0.2,
        label='Std Dev'
    )
    plt.xlabel('Training Steps')
    plt.ylabel('Episode Reward')
    plt.title(f'Training Progress: {title}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}_progress.png')
    plt.close()

def save_training_metrics(runner, filename):
    """Save training metrics for later analysis"""
    metrics = {
        'steps': runner.x_data,
        'rewards': runner.y_data,
        'reward_std': runner.y_dataerr,
        'training_time': (runner.times[-1] - runner.times[0]).total_seconds()
    }
    with open(filename, 'wb') as f:
        pickle.dump(metrics, f)

def setup_offscreen_rendering():
    """Configure environment for offscreen rendering"""
    import os
    os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for offscreen rendering
    # Fallback to software renderer if EGL is not available
    try:
        import mujoco
        _ = mujoco.GLContext(64, 64)  # Test EGL context creation
    except Exception:
        logger.info("EGL not available, falling back to osmesa")
        os.environ['MUJOCO_GL'] = 'osmesa'

def render_episode(runner, episode_frames, episode_num, output_dir="renders"):
    """Save episode frames as video"""
    import cv2
    import numpy as np
    from pathlib import Path

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Define the output path
    output_path = f"{output_dir}/episode_{episode_num}.mp4"
    
    if not episode_frames:
        logger.warning("No frames to render")
        return

    # Get frame dimensions
    height, width = episode_frames[0].shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    # Write frames
    for frame in episode_frames:
        # Convert from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    logger.info(f"Saved episode video to {output_path}")

#############################
# Flat Terrain Training Phase
#############################

def train_flat_terrain():
    """Train the initial policy on flat terrain"""
    logger.info("=" * 50)
    logger.info("Starting flat terrain training phase")
    logger.info("=" * 50)

    # Setup offscreen rendering
    setup_offscreen_rendering()

    # Initialize runner with flat terrain config
    args = create_training_args(task="flat_terrain", load_existing=False)
    logger.info("Training configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    runner = ZBotRunner(args, logger)

    # Train policy
    logger.info("Beginning training loop...")
    runner.train()

    # Log training statistics
    logger.info("Training completed. Final statistics:")
    logger.info(f"  Total steps: {len(runner.x_data)}")
    logger.info(f"  Final reward: {runner.y_data[-1]:.2f} ± {runner.y_dataerr[-1]:.2f}")
    logger.info(f"  Training time: {(runner.times[-1] - runner.times[0]).total_seconds():.2f}s")

    # Plot and save results
    logger.info("Saving training visualizations and metrics...")
    plot_training_progress(runner, "Flat Terrain Training")
    save_training_metrics(runner, "flat_terrain_metrics.pkl")

    # Evaluate and render using offscreen rendering
    try:
        logger.info("Starting flat terrain policy evaluation with offscreen rendering...")
        episode_frames = []
        
        def frame_callback(frame):
            episode_frames.append(frame)
        
        # Modify the evaluate method to collect frames
        runner.eval_env.set_render_callback(frame_callback)
        runner.evaluate()
        
        # Save the rendered episode
        render_episode(runner, episode_frames, episode_num=0)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)

    return runner

##############################
# Rough Terrain Training Phase
##############################

def train_with_curriculum(initial_runner):
    """Train with progressive terrain difficulty"""
    logger.info("=" * 50)
    logger.info("Starting curriculum training")
    logger.info("=" * 50)
    
    args = create_training_args(task="rough_terrain", load_existing=True)
    runner = ZBotRunner(args, logger)
    runner.params = initial_runner.params
    
    current_roughness = args.initial_roughness
    
    while current_roughness <= args.final_roughness:
        logger.info(f"Training at roughness level: {current_roughness:.2f}")
        
        # Update environment parameters
        runner.env.set_terrain_roughness(current_roughness)
        
        # Train for several episodes at current difficulty
        success_rate = runner.train_and_evaluate()
        
        # Advance curriculum if performance is good enough
        if success_rate >= args.min_success_rate:
            current_roughness += args.roughness_increment
            logger.info(f"Advancing to roughness level: {current_roughness:.2f}")
        else:
            logger.info("Performance below threshold, continuing at current level")
            
        # Save checkpoint
        runner.save_checkpoint(f"rough_terrain_r{current_roughness:.2f}.pkl")
    
    return runner

#######################
# Analysis & Evaluation
#######################

def analyze_performance(flat_metrics, rough_metrics):
    """Compare and analyze training performance"""
    logger.info("=" * 50)
    logger.info("Performance Analysis")
    logger.info("=" * 50)
    
    # Print summary statistics
    logger.info("Training Summary:")
    logger.info("Flat Terrain:")
    logger.info(f"  Training time: {flat_metrics['training_time']:.2f}s")
    logger.info(f"  Final reward: {flat_metrics['rewards'][-1]:.2f} ± {flat_metrics['reward_std'][-1]:.2f}")
    logger.info(f"  Peak reward: {max(flat_metrics['rewards']):.2f}")
    
    logger.info("Rough Terrain:")
    logger.info(f"  Training time: {rough_metrics['training_time']:.2f}s")
    logger.info(f"  Final reward: {rough_metrics['rewards'][-1]:.2f} ± {rough_metrics['reward_std'][-1]:.2f}")
    logger.info(f"  Peak reward: {max(rough_metrics['rewards']):.2f}")
    
    # Create comparison plot
    logger.info("Generating performance comparison plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot flat terrain progress
    plt.plot(flat_metrics['steps'], flat_metrics['rewards'], 
             label='Flat Terrain', color='blue')
    plt.fill_between(
        flat_metrics['steps'],
        np.array(flat_metrics['rewards']) - np.array(flat_metrics['reward_std']),
        np.array(flat_metrics['rewards']) + np.array(flat_metrics['reward_std']),
        alpha=0.2,
        color='blue'
    )
    
    # Plot rough terrain progress
    plt.plot(rough_metrics['steps'], rough_metrics['rewards'], 
             label='Rough Terrain', color='red')
    plt.fill_between(
        rough_metrics['steps'],
        np.array(rough_metrics['rewards']) - np.array(rough_metrics['reward_std']),
        np.array(rough_metrics['rewards']) + np.array(rough_metrics['reward_std']),
        alpha=0.2,
        color='red'
    )
    
    plt.xlabel('Training Steps')
    plt.ylabel('Episode Reward')
    plt.title('Training Progress Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_comparison.png')
    plt.close()

##############
# Main Script
##############

def main():
    """Main training pipeline"""
    logger.info("=" * 50)
    logger.info("Starting ZBot Training Pipeline")
    logger.info("=" * 50)
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Train on flat terrain
        logger.info("Starting flat terrain training phase...")
        flat_runner = train_flat_terrain()
        
        # Train on rough terrain
        logger.info("Starting rough terrain adaptation phase...")
        rough_runner = train_with_curriculum(flat_runner)
        
        # Load and analyze results
        logger.info("Loading training metrics for analysis...")
        with open("flat_terrain_metrics.pkl", 'rb') as f:
            flat_metrics = pickle.load(f)
        with open("rough_terrain_metrics.pkl", 'rb') as f:
            rough_metrics = pickle.load(f)
        
        analyze_performance(flat_metrics, rough_metrics)
        
        logger.info("Training pipeline completed successfully!")
        logger.info("Check the outputs directory for results and visualizations.")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 