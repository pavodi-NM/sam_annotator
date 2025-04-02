# tests/test_memory_manager.py

import os
import sys
import torch
import time
import logging
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sam_annotator.core.memory_manager import GPUMemoryManager



def format_bytes(bytes_num: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024
    return f"{bytes_num:.2f} PB"

def format_memory_info(info: dict) -> str:
    """Format memory info into human readable string."""
    used = format_bytes(info['used'])
    total = format_bytes(info['total'])
    percentage = info['utilization'] * 100
    return f"Used: {used} / Total: {total} ({percentage:.1f}%)"

def test_memory_allocation(memory_manager, logger):
    """Test allocating and freeing memory."""
    try:
        # Initial memory state
        initial_info = memory_manager.get_gpu_memory_info()
        logger.info(f"Initial GPU memory state: {format_memory_info(initial_info)}")

        # Allocate some tensors
        tensors = []
        for i in range(5):
            # Allocate a 1GB tensor
            size = 256 * 1024 * 1024  # ~1GB
            tensor = torch.zeros(size, device='cuda')
            tensors.append(tensor)
            
            # Check memory status
            status_ok, message = memory_manager.check_memory_status()
            current_info = memory_manager.get_gpu_memory_info()
            logger.info(f"After allocation {i+1}: {format_memory_info(current_info)}")
            if message:
                logger.warning(message)
                
            # If we hit critical threshold, break
            if not status_ok:
                logger.warning("Hit critical memory threshold!")
                break
                
            time.sleep(1)

        # Try to optimize memory
        logger.info("Attempting memory optimization...")
        memory_manager.optimize_memory(force=True)
        
        # Check memory after optimization
        post_opt_info = memory_manager.get_gpu_memory_info()
        logger.info(f"After optimization: {format_memory_info(post_opt_info)}")

    except Exception as e:
        logger.error(f"Error during memory test: {e}")

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Get memory fractions from environment or use defaults
    memory_fractions = [0.9, 0.7, 0.5]  # defaults
    if os.getenv('TEST_MEMORY_FRACTIONS'):
        try:
            memory_fractions = [float(x) for x in 
                              os.getenv('TEST_MEMORY_FRACTIONS').split(',')]
        except ValueError:
            logger.warning("Invalid TEST_MEMORY_FRACTIONS format, using defaults")
    
    for fraction in memory_fractions:
        logger.info(f"\nTesting with memory fraction: {fraction}")
        
        # Set environment variable
        os.environ['SAM_GPU_MEMORY_FRACTION'] = str(fraction)
        
        # Create memory manager with new settings
        memory_manager = GPUMemoryManager()
        
        # Run test
        test_memory_allocation(memory_manager, logger)
        
        # Cleanup
        torch.cuda.empty_cache()
        time.sleep(2)

if __name__ == "__main__":
    main()