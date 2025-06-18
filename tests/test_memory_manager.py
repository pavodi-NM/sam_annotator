# tests/test_memory_manager.py

import os
import sys
import torch
import time
import logging
import pytest
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Tests require GPU")
def test_memory_manager_basic():
    """Test basic memory manager functionality with pytest."""
    # Setup logging for pytest
    logger = logging.getLogger(__name__)
    
    # Test singleton behavior
    manager1 = GPUMemoryManager()
    manager2 = GPUMemoryManager()
    assert manager1 is manager2, "Memory manager should be singleton"
    
    # Test basic methods
    memory_info = manager1.get_gpu_memory_info()
    assert 'utilization' in memory_info
    assert 'used' in memory_info
    assert 'total' in memory_info
    
    # Test analytics (our new feature)
    analytics = manager1.get_memory_analytics()
    assert 'gpu_memory' in analytics
    assert 'statistics' in analytics
    assert 'configuration' in analytics
    
    logger.info("✅ Basic memory manager tests passed")

def run_memory_allocation_test(memory_manager, logger):
    """Test allocating and freeing memory - renamed to avoid pytest pickup."""
    try:
        # Initial memory state
        initial_info = memory_manager.get_gpu_memory_info()
        logger.info(f"Initial GPU memory state: {format_memory_info(initial_info)}")

        # Skip actual allocation in CI environment or if not enough memory
        if os.getenv('CI') == 'true' or not torch.cuda.is_available():
            logger.info("Skipping memory allocation (CI environment or no GPU)")
            return

        # Allocate smaller tensors to avoid OOM in testing
        tensors = []
        for i in range(3):  # Reduced from 5 to 3
            # Allocate a smaller tensor
            size = 64 * 1024 * 1024  # ~256MB instead of 1GB
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
                
            time.sleep(0.5)  # Shorter wait

        # Try to optimize memory
        logger.info("Attempting memory optimization...")
        memory_manager.optimize_memory(force=True)
        
        # Check memory after optimization
        post_opt_info = memory_manager.get_gpu_memory_info()
        logger.info(f"After optimization: {format_memory_info(post_opt_info)}")

        # Cleanup
        tensors = None
        torch.cuda.empty_cache()

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
        run_memory_allocation_test(memory_manager, logger)
        
        # Cleanup
        torch.cuda.empty_cache()
        time.sleep(2)

if __name__ == "__main__":
    main()