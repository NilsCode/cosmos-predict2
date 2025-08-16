# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to convert LeRobot datasets to Cosmos video2world format.

This script processes downloaded LeRobot datasets and converts them to the format expected by Cosmos:
- Creates metas/ folder with .txt files containing prompts
- Creates videos/ folder with .mp4 files
- Parses episodes.jsonl to extract task descriptions for each episode
- Supports combining multiple datasets into a single output folder with unique naming

Usage:
    # Convert a single dataset
    python scripts/convert_lerobot_to_cosmos.py --dataset_path datasets/youliangtan_so101-table-cleanup --camera_view front

    # Convert multiple datasets from a config file (individual outputs)
    python scripts/convert_lerobot_to_cosmos.py --config_file datasets_config.yaml --max_datasets 10 --camera_view front

    # Convert selected datasets from a YAML config into a combined dataset
    python scripts/convert_lerobot_to_cosmos.py --selection_config selection_config.yaml
    
    # Override output folder from command line (optional)
    python scripts/convert_lerobot_to_cosmos.py --selection_config selection_config.yaml --combined_output custom_output_folder

    # List available camera views for a dataset
    python scripts/convert_lerobot_to_cosmos.py --dataset_path datasets/youliangtan_so101-table-cleanup --list_cameras
"""

import argparse
import json
import os
import shutil
import uuid
import yaml
from pathlib import Path
from typing import List, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LeRobot datasets to Cosmos video2world format")
    
    # Single dataset conversion
    parser.add_argument("--dataset_path", type=str, help="Path to a single dataset to convert")
    
    # Batch conversion from config
    parser.add_argument("--config_file", type=str, help="Path to datasets config YAML file")
    parser.add_argument("--max_datasets", type=int, default=None, help="Maximum number of datasets to convert")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for dataset conversion")
    
    # Combined dataset conversion from selection config
    parser.add_argument("--selection_config", type=str, help="Path to YAML config file specifying datasets to combine")
    parser.add_argument("--combined_output", type=str, default="cosmos_predict2_video2world_dataset", 
                       help="Output directory name for combined dataset (overrides output_folder in selection config)")
    
    # Camera selection
    parser.add_argument("--camera_view", type=str, default="front", 
                       help="Camera view to use (e.g., 'front', 'wrist'). Use --list_cameras to see available options")
    parser.add_argument("--list_cameras", action="store_true", 
                       help="List available camera views for the dataset and exit")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="datasets", 
                       help="Base output directory for converted datasets")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing converted datasets")
    parser.add_argument("--dry_run", action="store_true", 
                       help="Show what would be done without actually converting")
    
    return parser.parse_args()


def load_episodes_data(dataset_path: str) -> List[Dict]:
    """Load and parse episodes.jsonl file."""
    episodes_file = os.path.join(dataset_path, "meta", "episodes.jsonl")
    
    if not os.path.exists(episodes_file):
        raise FileNotFoundError(f"episodes.jsonl not found at {episodes_file}")
    
    episodes = []
    with open(episodes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    
    return episodes


def find_available_cameras(dataset_path: str) -> List[str]:
    """Find available camera views in the dataset."""
    videos_dir = os.path.join(dataset_path, "videos")
    if not os.path.exists(videos_dir):
        return []
    
    # Look for camera directories in chunk-000
    chunk_dirs = [d for d in os.listdir(videos_dir) if d.startswith("chunk-")]
    if not chunk_dirs:
        return []
    
    first_chunk = os.path.join(videos_dir, chunk_dirs[0])
    camera_dirs = []
    
    for item in os.listdir(first_chunk):
        item_path = os.path.join(first_chunk, item)
        if os.path.isdir(item_path) and "observation.images." in item:
            camera_name = item.replace("observation.images.", "")
            camera_dirs.append(camera_name)
    
    return sorted(camera_dirs)


def get_video_files_for_camera(dataset_path: str, camera_view: str) -> Dict[int, str]:
    """Get mapping of episode index to video file path for a specific camera."""
    videos_dir = os.path.join(dataset_path, "videos")
    video_files = {}
    
    # Look through all chunks
    for chunk_dir in os.listdir(videos_dir):
        if not chunk_dir.startswith("chunk-"):
            continue
            
        camera_dir = os.path.join(videos_dir, chunk_dir, f"observation.images.{camera_view}")
        if not os.path.exists(camera_dir):
            continue
        
        for video_file in os.listdir(camera_dir):
            if video_file.endswith(".mp4") and video_file.startswith("episode_"):
                # Extract episode number from filename
                episode_num = int(video_file.replace("episode_", "").replace(".mp4", ""))
                video_files[episode_num] = os.path.join(camera_dir, video_file)
    
    return video_files


def convert_dataset(dataset_path: str, camera_view: str, output_dir: str, 
                   overwrite: bool = False, dry_run: bool = False) -> bool:
    """Convert a single dataset to Cosmos format."""
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    output_path = os.path.join(output_dir, f"{dataset_name}_cosmos")
    
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Input path: {dataset_path}")
    print(f"Output path: {output_path}")
    
    # Check if output already exists
    if os.path.exists(output_path) and not overwrite:
        print(f"  ⚠️  Output directory already exists: {output_path}")
        print("     Use --overwrite to replace existing data")
        return False
    
    # Load episodes data
    try:
        episodes = load_episodes_data(dataset_path)
        print(f"  📄 Found {len(episodes)} episodes")
    except FileNotFoundError as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # Check camera availability
    available_cameras = find_available_cameras(dataset_path)
    if not available_cameras:
        print(f"  ❌ No camera views found in dataset")
        return False
    
    print(f"  📹 Available cameras: {', '.join(available_cameras)}")
    
    if camera_view not in available_cameras:
        print(f"  ❌ Camera view '{camera_view}' not available")
        print(f"     Available options: {', '.join(available_cameras)}")
        return False
    
    # Get video files for the selected camera
    video_files = get_video_files_for_camera(dataset_path, camera_view)
    print(f"  🎥 Found {len(video_files)} video files for camera '{camera_view}'")
    
    if dry_run:
        if len(episodes) == 1 and len(video_files) > 1:
            print(f"  🔍 DRY RUN: Would create {len(video_files)} prompt files (single task) and copy {len(video_files)} videos")
        else:
            print(f"  🔍 DRY RUN: Would create {len(episodes)} prompt files and copy {len(video_files)} videos")
        return True
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    metas_dir = os.path.join(output_path, "metas")
    videos_dir = os.path.join(output_path, "videos")
    os.makedirs(metas_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    # Check if we have a single task for multiple videos case
    if len(episodes) == 1 and len(video_files) > 1:
        print(f"  ℹ️  Single task applies to {len(video_files)} videos")
    
    # Handle two cases:
    # Case 1: One episode entry per video (normal case)
    # Case 2: One episode entry for multiple videos (single task for all)
    converted_count = 0
    
    if len(episodes) == 1 and len(video_files) > 1:
        # Case 2: Single task for multiple videos
        episode = episodes[0]
        tasks = episode["tasks"]
        prompt = tasks[0] if tasks else "Robot manipulation task"
        
        # Process all video files with the same task description
        for video_idx, video_path in video_files.items():
            # Create prompt file
            prompt_filename = f"episode_{video_idx:06d}.txt"
            prompt_path = os.path.join(metas_dir, prompt_filename)
            
            with open(prompt_path, 'w') as f:
                f.write(prompt)
            
            # Copy video file
            dst_video = os.path.join(videos_dir, f"episode_{video_idx:06d}.mp4")
            
            try:
                shutil.copy2(video_path, dst_video)
                converted_count += 1
            except Exception as e:
                print(f"  ⚠️  Failed to copy video for episode {video_idx}: {e}")
    
    else:
        # Case 1: Normal case - one episode entry per video
        for episode in episodes:
            episode_idx = episode["episode_index"]
            tasks = episode["tasks"]
            
            # Use the first task as the prompt (most episodes have single tasks)
            prompt = tasks[0] if tasks else f"Episode {episode_idx}"
            
            # Create prompt file
            prompt_filename = f"episode_{episode_idx:06d}.txt"
            prompt_path = os.path.join(metas_dir, prompt_filename)
            
            with open(prompt_path, 'w') as f:
                f.write(prompt)
            
            # Copy video file if it exists
            if episode_idx in video_files:
                src_video = video_files[episode_idx]
                dst_video = os.path.join(videos_dir, f"episode_{episode_idx:06d}.mp4")
                
                try:
                    shutil.copy2(src_video, dst_video)
                    converted_count += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to copy video for episode {episode_idx}: {e}")
            else:
                print(f"  ⚠️  No video file found for episode {episode_idx}")
    
    print(f"  ✅ Successfully converted {converted_count} episodes")
    print(f"     Prompts saved to: {metas_dir}")
    print(f"     Videos saved to: {videos_dir}")
    
    return True


def load_datasets_config(config_file: str) -> List[str]:
    """Load dataset names from config file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_names = []
    for dataset in config.get('datasets', []):
        if 'name' in dataset:
            dataset_names.append(dataset['name'])
    
    return dataset_names


def load_selection_config(config_file: str) -> Dict:
    """Load selection config file specifying datasets to combine."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def convert_dataset_to_combined(dataset_path: str, dataset_name: str, camera_view: str, 
                               combined_metas_dir: str, combined_videos_dir: str, 
                               dry_run: bool = False) -> int:
    """Convert a single dataset and add to combined output with unique naming."""
    print(f"\n📦 Processing dataset: {dataset_name}")
    print(f"   Input path: {dataset_path}")
    
    # Load episodes data
    try:
        episodes = load_episodes_data(dataset_path)
        print(f"   📄 Found {len(episodes)} episode entries")
    except FileNotFoundError as e:
        print(f"   ❌ Error: {e}")
        return 0
    
    # Check camera availability
    available_cameras = find_available_cameras(dataset_path)
    if not available_cameras:
        print(f"   ❌ No camera views found in dataset")
        return 0
    
    print(f"   📹 Available cameras: {', '.join(available_cameras)}")
    
    if camera_view not in available_cameras:
        print(f"   ❌ Camera view '{camera_view}' not available")
        print(f"      Available options: {', '.join(available_cameras)}")
        return 0
    
    # Get video files for the selected camera
    video_files = get_video_files_for_camera(dataset_path, camera_view)
    print(f"   🎥 Found {len(video_files)} video files for camera '{camera_view}'")
    
    # Check if we have a single task for multiple videos case
    if len(episodes) == 1 and len(video_files) > 1:
        print(f"   ℹ️  Single task applies to {len(video_files)} videos")
    
    if dry_run:
        print(f"   🔍 DRY RUN: Would add {len(video_files)} files to combined dataset")
        return len(video_files)
    
    # Handle two cases:
    # Case 1: One episode entry per video (normal case)
    # Case 2: One episode entry for multiple videos (single task for all)
    converted_count = 0
    
    if len(episodes) == 1 and len(video_files) > 1:
        # Case 2: Single task for multiple videos
        episode = episodes[0]
        tasks = episode["tasks"]
        prompt = tasks[0] if tasks else "Robot manipulation task"
        
        # Process all video files with the same task description
        for video_idx, video_path in video_files.items():
            # Generate unique filename with dataset name and UUID
            unique_id = str(uuid.uuid4())[:8]
            safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_')
            base_filename = f"{safe_dataset_name}_ep{video_idx:06d}_{unique_id}"
            
            # Create prompt file
            prompt_filename = f"{base_filename}.txt"
            prompt_path = os.path.join(combined_metas_dir, prompt_filename)
            
            with open(prompt_path, 'w') as f:
                f.write(prompt)
            
            # Copy video file
            dst_video = os.path.join(combined_videos_dir, f"{base_filename}.mp4")
            
            try:
                shutil.copy2(video_path, dst_video)
                converted_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to copy video for episode {video_idx}: {e}")
    
    else:
        # Case 1: Normal case - one episode entry per video
        for episode in episodes:
            episode_idx = episode["episode_index"]
            tasks = episode["tasks"]
            
            # Use the first task as the prompt (most episodes have single tasks)
            prompt = tasks[0] if tasks else f"Episode {episode_idx}"
            
            # Generate unique filename with dataset name and UUID
            unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
            safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_')
            base_filename = f"{safe_dataset_name}_ep{episode_idx:06d}_{unique_id}"
            
            # Create prompt file
            prompt_filename = f"{base_filename}.txt"
            prompt_path = os.path.join(combined_metas_dir, prompt_filename)
            
            with open(prompt_path, 'w') as f:
                f.write(prompt)
            
            # Copy video file if it exists
            if episode_idx in video_files:
                src_video = video_files[episode_idx]
                dst_video = os.path.join(combined_videos_dir, f"{base_filename}.mp4")
                
                try:
                    shutil.copy2(src_video, dst_video)
                    converted_count += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to copy video for episode {episode_idx}: {e}")
            else:
                print(f"   ⚠️  No video file found for episode {episode_idx}")
    
    print(f"   ✅ Successfully added {converted_count} episodes to combined dataset")
    return converted_count


def main():
    args = parse_args()
    
    if args.dataset_path:
        # Single dataset conversion
        if args.list_cameras:
            cameras = find_available_cameras(args.dataset_path)
            if cameras:
                print(f"Available camera views for {os.path.basename(args.dataset_path)}:")
                for camera in cameras:
                    print(f"  - {camera}")
            else:
                print("No camera views found in dataset")
            return
        
        success = convert_dataset(
            args.dataset_path, 
            args.camera_view, 
            args.output_dir,
            args.overwrite,
            args.dry_run
        )
        
        if success and not args.dry_run:
            print(f"\n🎉 Conversion completed successfully!")
            print(f"To generate T5 embeddings, run:")
            output_path = os.path.join(args.output_dir, f"{os.path.basename(args.dataset_path.rstrip('/'))}_cosmos")
            print(f"python -m scripts.get_t5_embeddings --dataset_path {output_path}")
    
    elif args.selection_config:
        # Combined dataset conversion from selection config
        if not os.path.exists(args.selection_config):
            print(f"❌ Selection config file not found: {args.selection_config}")
            return
        
        config = load_selection_config(args.selection_config)
        selected_datasets = config.get('selected_datasets', [])
        camera_view = config.get('camera_view', args.camera_view)
        
        # Check if combined_output was explicitly provided via command line
        import sys
        cmd_line_has_combined_output = '--combined_output' in sys.argv
        
        # Use command line argument if explicitly provided, otherwise use config file, otherwise use default
        if cmd_line_has_combined_output:
            output_folder_name = args.combined_output
        else:
            output_folder_name = config.get('output_folder', args.combined_output)
        
        print(f"📋 Found {len(selected_datasets)} datasets in selection config")
        print(f"📹 Using camera view: {camera_view}")
        print(f"📁 Output folder: {output_folder_name}")
        
        # Create combined output directory
        combined_output_path = os.path.join(args.output_dir, output_folder_name)
        
        if os.path.exists(combined_output_path) and not args.overwrite:
            print(f"⚠️  Combined output directory already exists: {combined_output_path}")
            print("   Use --overwrite to replace existing data")
            return
        
        if not args.dry_run:
            os.makedirs(combined_output_path, exist_ok=True)
            combined_metas_dir = os.path.join(combined_output_path, "metas")
            combined_videos_dir = os.path.join(combined_output_path, "videos")
            os.makedirs(combined_metas_dir, exist_ok=True)
            os.makedirs(combined_videos_dir, exist_ok=True)
        else:
            combined_metas_dir = combined_videos_dir = ""
        
        print(f"🎯 Combined output: {combined_output_path}")
        
        total_converted = 0
        successful_datasets = 0
        
        for dataset_info in selected_datasets:
            if isinstance(dataset_info, str):
                dataset_name = dataset_info
                dataset_camera = camera_view
            elif isinstance(dataset_info, dict):
                dataset_name = dataset_info.get('name')
                dataset_camera = dataset_info.get('camera_view', camera_view)
            else:
                print(f"⚠️  Invalid dataset entry: {dataset_info}")
                continue
            
            if not dataset_name:
                print(f"⚠️  Dataset name not found in entry: {dataset_info}")
                continue
            
            # Convert dataset name to directory name
            dataset_dir = dataset_name.replace('/', '_')
            dataset_path = os.path.join(args.output_dir, dataset_dir)
            
            if not os.path.exists(dataset_path):
                print(f"\n⚠️  Dataset {dataset_name} not found at {dataset_path}")
                continue
            
            converted_count = convert_dataset_to_combined(
                dataset_path,
                dataset_name,
                dataset_camera,
                combined_metas_dir,
                combined_videos_dir,
                args.dry_run
            )
            
            if converted_count > 0:
                total_converted += converted_count
                successful_datasets += 1
        
        print(f"\n🎉 Combined dataset creation completed!")
        print(f"Successfully processed {successful_datasets}/{len(selected_datasets)} datasets")
        print(f"Total episodes added: {total_converted}")
        
        if total_converted > 0 and not args.dry_run:
            print(f"\nCombined dataset saved to: {combined_output_path}")
            print(f"To generate T5 embeddings, run:")
            print(f"python -m scripts.get_t5_embeddings --dataset_path {combined_output_path}")
    
    elif args.config_file:
        # Batch conversion from config (individual outputs)
        if not os.path.exists(args.config_file):
            print(f"❌ Config file not found: {args.config_file}")
            return
        
        dataset_names = load_datasets_config(args.config_file)
        print(f"📋 Found {len(dataset_names)} datasets in config")
        
        if args.max_datasets:
            end_index = min(args.start_index + args.max_datasets, len(dataset_names))
            dataset_names = dataset_names[args.start_index:end_index]
            print(f"🔢 Processing datasets {args.start_index} to {end_index-1}")
        
        successful_conversions = 0
        
        for i, dataset_name in enumerate(dataset_names, args.start_index):
            # Convert dataset name to directory name (replace / with _)
            dataset_dir = dataset_name.replace('/', '_')
            dataset_path = os.path.join(args.output_dir, dataset_dir)
            
            if not os.path.exists(dataset_path):
                print(f"\n⚠️  Dataset {i}: {dataset_name} not found at {dataset_path}")
                continue
            
            print(f"\n📦 Dataset {i}: {dataset_name}")
            success = convert_dataset(
                dataset_path,
                args.camera_view,
                args.output_dir,
                args.overwrite,
                args.dry_run
            )
            
            if success:
                successful_conversions += 1
        
        print(f"\n🎉 Batch conversion completed!")
        print(f"Successfully converted {successful_conversions}/{len(dataset_names)} datasets")
        
        if successful_conversions > 0 and not args.dry_run:
            print(f"\nTo generate T5 embeddings for all converted datasets, you can run:")
            print(f"for dir in {args.output_dir}/*_cosmos; do")
            print(f"  echo \"Processing $dir\"")
            print(f"  python -m scripts.get_t5_embeddings --dataset_path \"$dir\"")
            print(f"done")
    
    else:
        print("❌ Please specify either --dataset_path, --config_file, or --selection_config")
        print("Use --help for usage information")


if __name__ == "__main__":
    main() 