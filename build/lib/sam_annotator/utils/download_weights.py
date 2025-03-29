#!/usr/bin/env python
"""
Utility script to download SAM model weights.
"""

import os
import argparse
import logging
from pathlib import Path
import urllib.request
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available model weights and their URLs
SAM1_WEIGHTS = {
    'vit_h': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'filename': 'sam_vit_h_4b8939.pth'
    },
    'vit_l': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'filename': 'sam_vit_l_0b3195.pth'
    },
    'vit_b': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'filename': 'sam_vit_b_01ec64.pth'
    }
}

# SAM2 weights
SAM2_WEIGHTS = {
    'small_v2': {
        'url': 'https://ai.meta.com/download/model_weights/segment_anything_2/sam2_s_vit_t_16_0a_2of12.pt',
        'filename': 'sam2_small_v2.pt'
    },
    'base_v2': {
        'url': 'https://ai.meta.com/download/model_weights/segment_anything_2/sam2_b_vit_b_36_0a_2of12.pt',
        'filename': 'sam2_base_v2.pt'
    }
}

def download_file(url, dest_path):
    """Download a file from URL to destination path with progress reporting."""
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            file_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            logger.info(f"Downloading {url} ({file_size/1024/1024:.1f} MB)")
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                progress = (downloaded / file_size) * 100 if file_size > 0 else 0
                logger.info(f"Progress: {progress:.1f}% ({downloaded/1024/1024:.1f} MB)")
        
        logger.info(f"Download completed: {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def main():
    """Main function to download SAM weights."""
    parser = argparse.ArgumentParser(description='Download SAM model weights')
    parser.add_argument('--sam_version', type=str, choices=['sam1', 'sam2', 'all'], default='all',
                       help='Which SAM version weights to download')
    parser.add_argument('--model_type', type=str, 
                       help='Specific model type to download. For SAM1: vit_h, vit_l, vit_b. '
                            'For SAM2: small_v2, base_v2')
    parser.add_argument('--output_dir', type=str, default='weights',
                       help='Directory to save model weights')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Weights will be saved to {output_dir.resolve()}")
    
    # Download specified weights
    if args.sam_version in ['sam1', 'all']:
        if args.model_type and args.model_type in SAM1_WEIGHTS:
            # Download specific SAM1 model
            weight_info = SAM1_WEIGHTS[args.model_type]
            dest_path = output_dir / weight_info['filename']
            logger.info(f"Downloading SAM1 {args.model_type} model...")
            download_file(weight_info['url'], dest_path)
        elif not args.model_type:
            # Download all SAM1 models
            logger.info("Downloading all SAM1 models...")
            for model_type, weight_info in SAM1_WEIGHTS.items():
                dest_path = output_dir / weight_info['filename']
                logger.info(f"Downloading SAM1 {model_type} model...")
                download_file(weight_info['url'], dest_path)
    
    if args.sam_version in ['sam2', 'all']:
        if args.model_type and args.model_type in SAM2_WEIGHTS:
            # Download specific SAM2 model
            weight_info = SAM2_WEIGHTS[args.model_type]
            dest_path = output_dir / weight_info['filename']
            logger.info(f"Downloading SAM2 {args.model_type} model...")
            download_file(weight_info['url'], dest_path)
        elif not args.model_type:
            # Download all SAM2 models
            logger.info("Downloading all SAM2 models...")
            for model_type, weight_info in SAM2_WEIGHTS.items():
                dest_path = output_dir / weight_info['filename']
                logger.info(f"Downloading SAM2 {model_type} model...")
                download_file(weight_info['url'], dest_path)
    
    logger.info("Download process completed!")

if __name__ == '__main__':
    main() 