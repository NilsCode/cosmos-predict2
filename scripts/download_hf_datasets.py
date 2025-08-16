#!/usr/bin/env python3
"""
Script to download HuggingFace datasets based on YAML configuration.

Features:
- Parallel video/LFS file and metadata downloads using ThreadPoolExecutor
- Progress tracking with tqdm
- Configurable number of parallel workers
- Thread-safe logging

Usage: python scripts/download_hf_datasets.py [--config datasets_config.yaml]

Performance improvements:
- Use --max-workers N to control parallelism (default: 4)
- Video and metadata downloads are now parallelized for faster completion
- Progress bars show real-time download status
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import yaml
from datasets import load_dataset
from huggingface_hub import login, HfApi, hf_hub_download

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

# Global lock for thread-safe printing
print_lock = Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def download_single_lfs_file(
    dataset_name: str,
    file_path: str,
    dataset_dir: Path,
    use_auth_token: bool = False
) -> tuple[str, bool]:
    """Download a single LFS file. Returns (file_path, success)."""
    try:
        local_file_path = dataset_dir / file_path
        ensure_directory(str(local_file_path.parent))
        
        downloaded_path = hf_hub_download(
            repo_id=dataset_name,
            filename=file_path,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            local_dir_use_symlinks=False,
            token=use_auth_token if use_auth_token else None
        )
        
        return file_path, True
        
    except Exception as e:
        thread_safe_print(f"  ✗ Error downloading {file_path}: {str(e)}")
        return file_path, False


def download_lfs_files(
    dataset_name: str,
    output_dir: str,
    file_patterns: List[str] = None,
    use_auth_token: bool = False,
    max_workers: int = 4
) -> bool:
    """Download LFS files (videos, large files) and metadata from HuggingFace dataset with parallel downloads."""
    
    if not file_patterns:
        file_patterns = ["videos/", ".mp4", ".avi", ".mov", ".webm", "meta/", ".json"]
    
    thread_safe_print(f"Downloading LFS files for: {dataset_name}")
    
    try:
        api = HfApi()
        
        # List all files in the repository
        files = api.list_repo_files(dataset_name, repo_type="dataset")
        
        # Filter files that match our patterns
        lfs_files = []
        for file in files:
            if any(pattern in file for pattern in file_patterns):
                lfs_files.append(file)
        
        if not lfs_files:
            thread_safe_print(f"  No LFS files found matching patterns: {file_patterns}")
            return True
            
        thread_safe_print(f"  Found {len(lfs_files)} LFS files to download")
        thread_safe_print(f"  Using {max_workers} parallel workers")
        
        # Create dataset-specific directory
        dataset_dir = Path(output_dir) / dataset_name.replace("/", "_")
        ensure_directory(str(dataset_dir))
        
        # Download files in parallel with progress tracking
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_file = {
                executor.submit(
                    download_single_lfs_file,
                    dataset_name,
                    file_path,
                    dataset_dir,
                    use_auth_token
                ): file_path
                for file_path in lfs_files
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(lfs_files), desc="  Downloading", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        downloaded_file, success = future.result()
                        if success:
                            success_count += 1
                            pbar.set_postfix({"✓": success_count, "✗": len(lfs_files) - success_count})
                        pbar.update(1)
                    except Exception as e:
                        thread_safe_print(f"  ✗ Unexpected error with {file_path}: {str(e)}")
                        pbar.update(1)
        
        thread_safe_print(f"  ✓ Downloaded {success_count}/{len(lfs_files)} LFS files")
        return success_count > 0
        
    except Exception as e:
        thread_safe_print(f"✗ Error downloading LFS files for {dataset_name}: {str(e)}")
        return False


def download_dataset(
    dataset_name: str,
    output_dir: str,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    download_all_splits: bool = True,
    download_videos: bool = False,
    max_workers: int = 4
) -> None:
    """Download a single dataset from HuggingFace."""
    
    thread_safe_print(f"Downloading dataset: {dataset_name}")
    
    try:
        # Prepare arguments for load_dataset
        load_args = {
            "path": dataset_name,
            "cache_dir": cache_dir,
        }
        
        if subset:
            load_args["name"] = subset
            
        if not download_all_splits and split:
            load_args["split"] = split
            
        if use_auth_token:
            load_args["token"] = True
        
        # Load the dataset
        dataset = load_dataset(**load_args)
        
        # Create dataset-specific directory
        dataset_dir = Path(output_dir) / dataset_name.replace("/", "_")
        ensure_directory(str(dataset_dir))
        
        # Save dataset to disk
        if hasattr(dataset, 'save_to_disk'):
            # DatasetDict (multiple splits)
            dataset.save_to_disk(str(dataset_dir))
            thread_safe_print(f"✓ Saved dataset to: {dataset_dir}")
        else:
            # Single dataset
            dataset.save_to_disk(str(dataset_dir))
            thread_safe_print(f"✓ Saved dataset to: {dataset_dir}")
        
        # Download LFS files (videos and metadata) if requested
        if download_videos:
            thread_safe_print(f"Downloading videos and metadata for: {dataset_name}")
            video_success = download_lfs_files(
                dataset_name=dataset_name,
                output_dir=output_dir,
                file_patterns=["videos/", ".mp4", ".avi", ".mov", ".webm", "meta/", ".json"],
                use_auth_token=use_auth_token,
                max_workers=max_workers
            )
            if video_success:
                thread_safe_print(f"✓ Video and metadata download completed for: {dataset_name}")
            else:
                thread_safe_print(f"! Video and metadata download failed for: {dataset_name}")
            
    except Exception as e:
        thread_safe_print(f"✗ Error downloading {dataset_name}: {str(e)}")
        return False
        
    return True


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets from YAML config")
    parser.add_argument(
        "--config", 
        default="datasets_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Login to HuggingFace before downloading (for private datasets)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of datasets to download (downloads first N datasets from config)"
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download video files and metadata (LFS files) in addition to dataset data"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for video and metadata downloads (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please create a YAML config file with your dataset specifications.")
        sys.exit(1)
    
    # Login to HuggingFace if requested
    if args.login:
        print("Logging into HuggingFace...")
        login()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    # Extract configuration
    datasets_config = config.get("datasets", [])
    global_config = config.get("config", {})
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        datasets_config = datasets_config[:args.limit]
        print(f"Limiting download to first {args.limit} datasets")
    
    output_dir = global_config.get("output_dir", "./datasets")
    use_auth_token = global_config.get("use_auth_token", False)
    cache_dir = global_config.get("cache_dir")
    download_all_splits = global_config.get("download_all_splits", True)
    download_videos = global_config.get("download_videos", False) or args.download_videos
    max_workers = global_config.get("max_workers", args.max_workers)
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    print(f"Output directory: {output_dir}")
    print(f"Number of datasets to download: {len(datasets_config)}")
    print(f"Download videos and metadata: {'Yes' if download_videos else 'No'}")
    print(f"Max parallel workers: {max_workers}")
    print("-" * 50)
    
    # Download each dataset
    success_count = 0
    total_count = len(datasets_config)
    
    for dataset_config in datasets_config:
        if isinstance(dataset_config, str):
            # Simple string format
            dataset_name = dataset_config
            split = None
            subset = None
        elif isinstance(dataset_config, dict):
            # Dictionary format
            dataset_name = dataset_config.get("name")
            split = dataset_config.get("split")
            subset = dataset_config.get("subset")
        else:
            print(f"✗ Invalid dataset configuration: {dataset_config}")
            continue
        
        if not dataset_name:
            print("✗ Dataset name is required")
            continue
        
        success = download_dataset(
            dataset_name=dataset_name,
            output_dir=output_dir,
            split=split,
            subset=subset,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            download_all_splits=download_all_splits,
            download_videos=download_videos,
            max_workers=max_workers
        )
        
        if success:
            success_count += 1
        
        print("-" * 50)
    
    print(f"Download complete: {success_count}/{total_count} datasets downloaded successfully.")


if __name__ == "__main__":
    main() 