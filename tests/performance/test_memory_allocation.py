import os
import torch
import time
import logging
import pytest
from sam_annotator.core.memory_manager import GPUMemoryManager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Tests require GPU")
@pytest.mark.performance
def test_memory_allocation():
    """Test allocating and freeing memory."""
    logger = setup_logging()
    memory_manager = GPUMemoryManager()
    
    try:
        # Initial memory state
        initial_info = memory_manager.get_gpu_memory_info()
        logger.info(f"Initial memory state: {initial_info['formatted'] if 'formatted' in initial_info else initial_info}")

        # Skip actual allocation in CI environment
        if os.getenv('CI') == 'true':
            logger.info("Skipping memory allocation in CI environment")
            return
            
        # Allocate some tensors if GPU is available
        if torch.cuda.is_available():
            tensors = []
            for i in range(3):  # Reduced from 5 to 3
                # Allocate a smaller tensor to avoid OOM
                size = 64 * 1024 * 1024  # ~256MB 
                tensor = torch.zeros(size, device='cuda')
                tensors.append(tensor)
                
                # Check memory status
                status, message = memory_manager.check_memory_status()
                current_info = memory_manager.get_gpu_memory_info()
                logger.info(f"After allocation {i+1}: {current_info['formatted'] if 'formatted' in current_info else current_info}")
                
                # If we hit critical threshold, break
                if not status:
                    logger.warning(f"Hit critical memory threshold! {message}")
                    break
                    
                time.sleep(0.5)  # Shorter wait time

            # Check memory usage
            logger.info("Checking current memory usage...")
            # No log_memory_stats method in current class - use safe_get_memory_info instead
            memory_info = memory_manager.safe_get_memory_info()
            logger.info(f"Current memory usage: {memory_info['formatted']}")
            
            # Clean up
            tensors = None
            torch.cuda.empty_cache()
            
            # Check memory after cleanup
            post_cleanup_info = memory_manager.get_gpu_memory_info()
            logger.info(f"After cleanup: {post_cleanup_info['formatted'] if 'formatted' in post_cleanup_info else post_cleanup_info}")

    except Exception as e:
        logger.error(f"Error during memory test: {e}")
        
    # Final cleanup
    torch.cuda.empty_cache()

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
        test_memory_allocation()
        
        # Cleanup
        torch.cuda.empty_cache()
        time.sleep(2)  # Wait for cleanup

if __name__ == "__main__":
    main()