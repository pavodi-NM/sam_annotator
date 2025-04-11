import os
import torch
import time
import logging
from ..src.core.memory_manager_test import GPUMemoryManager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_memory_allocation(memory_manager, logger):
    """Test allocating and freeing memory."""
    try:
        # Initial memory state
        initial_info = memory_manager.get_gpu_memory_info()
        logger.info(f"Initial GPU memory state: {initial_info}")

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
            logger.info(f"After allocation {i+1}: {current_info}")
            if message:
                logger.warning(message)
                
            # If we hit critical threshold, break
            if not status_ok:
                logger.warning("Hit critical memory threshold!")
                break
                
            time.sleep(1)  # Wait to see the changes

        # Try to optimize memory
        logger.info("Attempting memory optimization...")
        memory_manager.optimize_memory(force=True)
        
        # Check memory after optimization
        post_opt_info = memory_manager.get_gpu_memory_info()
        logger.info(f"After optimization: {post_opt_info}")

    except Exception as e:
        logger.error(f"Error during memory test: {e}")

def main():
    logger = setup_logging()
    
    # Test with different memory fractions
    memory_fractions = [0.9, 0.7, 0.5]
    
    for fraction in memory_fractions:
        logger.info(f"\nTesting with memory fraction: {fraction}")
        
        # Set environment variable
        os.environ['SAM_GPU_MEMORY_FRACTION'] = str(fraction)
        os.environ['SAM_MEMORY_WARNING_THRESHOLD'] = '0.7'
        os.environ['SAM_MEMORY_CRITICAL_THRESHOLD'] = '0.9'
        
        # Create memory manager with new settings
        memory_manager = GPUMemoryManager()
        
        # Run test
        test_memory_allocation(memory_manager, logger)
        
        # Cleanup
        torch.cuda.empty_cache()
        time.sleep(2)  # Wait for cleanup

if __name__ == "__main__":
    main()