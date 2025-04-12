# Memory Management in SAM Annotator

This document provides a detailed overview of how SAM Annotator manages memory across different operating systems (Linux and Windows) and the caching systems implemented to enhance performance.

## Table of Contents

1. [Memory Management Architecture](#memory-management-architecture)
2. [Cross-Platform Memory Management](#cross-platform-memory-management)
   - [Linux-Based Systems](#linux-based-systems)
   - [Windows-Based Systems](#windows-based-systems)
3. [GPU Memory Management](#gpu-memory-management)
   - [Configuration Options](#configuration-options)
   - [Memory Manager Implementation](#memory-manager-implementation)
4. [Caching Systems](#caching-systems)
   - [Image Processing Cache](#image-processing-cache)
   - [Prediction Cache](#prediction-cache)
5. [Memory Optimization Strategies](#memory-optimization-strategies)
6. [Real-Time Memory Monitoring](#real-time-memory-monitoring)
7. [Troubleshooting Memory Issues](#troubleshooting-memory-issues)
8. [Testing Memory Management](#testing-memory-management)

## Memory Management Architecture

SAM Annotator uses a layered approach to memory management:

1. **GPU Memory Manager**: Central component that interfaces with NVIDIA SMI or PyTorch to monitor and manage GPU memory.
2. **Image Processor Cache**: Caches processed images to avoid redundant transformations.
3. **Prediction Cache**: Stores recent segmentation predictions to avoid redundant computations.
4. **Memory Optimizers**: Active memory management components that clear caches when memory usage exceeds thresholds.

## Cross-Platform Memory Management

### Linux-Based Systems

On Linux systems, SAM Annotator takes advantage of more advanced memory management capabilities:

- **NVIDIA SMI Integration**: When available, uses NVIDIA System Management Interface to get detailed GPU memory statistics.
- **Enhanced Memory Monitoring**: Accesses detailed memory utilization metrics including total, used, and free memory.
- **Background Process Optimization**: Linux's better handling of background processes allows for more aggressive caching strategies.

Example memory information on Linux:
```
Memory before prediction: Used: 2.45 GB / Total: 8.00 GB (30.6%)
```

### Windows-Based Systems

On Windows systems, SAM Annotator implements fallback mechanisms:

- **PyTorch Fallbacks**: When NVIDIA SMI is not available, falls back to PyTorch's memory management functions.
- **Simplified Memory Reporting**: Uses a more robust approach to memory reporting to handle potential missing metrics.
- **Safety Mechanisms**: Implements additional safety checks to prevent KeyError issues when certain memory metrics are not available.
- **Robust Error Handling**: The `safe_get_memory_info()` method guarantees a valid memory information dictionary, even on Windows or CPU-only systems.

Example memory information on Windows:
```
Memory before prediction: Running on CPU - memory stats not available
```

## GPU Memory Management

### Configuration Options

SAM Annotator's memory management is highly configurable through environment variables:

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `SAM_GPU_MEMORY_FRACTION` | 0.9 | Maximum fraction of GPU memory to use (0.0-1.0) |
| `SAM_MEMORY_WARNING_THRESHOLD` | 0.8 | Memory utilization threshold for warnings (0.0-1.0) |
| `SAM_MEMORY_CRITICAL_THRESHOLD` | 0.95 | Memory utilization threshold for critical errors (0.0-1.0) |
| `SAM_ENABLE_MEMORY_GROWTH` | True | Whether to allow dynamic memory growth or enforce the limit strictly |

### Memory Manager Implementation

The GPU memory management is centralized in the `GPUMemoryManager` class:

```python
class GPUMemoryManager:
    """Enhanced GPU memory manager with fallback options."""
    
    def __init__(self):
        # Load configuration from environment variables with defaults
        self.memory_fraction = self._get_env_float('SAM_GPU_MEMORY_FRACTION', 0.9)
        self.warning_threshold = self._get_env_float('SAM_MEMORY_WARNING_THRESHOLD', 0.8)
        self.critical_threshold = self._get_env_float('SAM_MEMORY_CRITICAL_THRESHOLD', 0.95)
        self.enable_memory_growth = self._get_env_bool('SAM_ENABLE_MEMORY_GROWTH', True)
        
        # Try to initialize NVIDIA SMI with fallbacks
        self.nvml_initialized = False
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            self.nvidia_smi = nvidia_smi
            self.nvml_initialized = True
        except ImportError:
            # Fallback to torch memory management
            pass
```

Key methods:
- `get_gpu_memory_info()`: Retrieves current memory statistics with platform-specific handling.
- `safe_get_memory_info()`: Guarantees a valid memory info dictionary, even on Windows/CPU systems.
- `check_memory_status()`: Evaluates memory usage against thresholds and returns status and warnings.
- `optimize_memory()`: Forces garbage collection and cache clearing when needed.
- `should_cache()`: Determines if it's safe to cache based on current memory usage.

The memory manager is initialized by both the `SAM1Predictor` and `SAM2Predictor` classes, and is used throughout the application for memory monitoring and optimization.

## Caching Systems

SAM Annotator employs multiple caching systems to optimize performance.

### Image Processing Cache

Implemented in `ImageProcessor` class, this cache reduces the computational overhead of resizing and preprocessing images:

```python
# Cache initialization in ImageProcessor.__init__
self._processed_cache = WeakValueDictionary()  # Cache processed images
self._metadata_cache = {}  # Cache metadata
self.max_cache_size = 10
```

Features:
- **Weak References**: Uses `WeakValueDictionary` to allow cached images to be garbage collected when memory is low.
- **Size-Limited Cache**: Maintains a maximum of 10 processed images by default.
- **Hash-Based Keys**: Uses MD5 hashes of image data to identify cached entries.
- **Memory Usage Monitoring**: Provides `get_memory_usage()` method to estimate cache size in bytes.
- **Explicit Cleanup**: Offers `clear_cache()` method for forced cleanup.

Implementation details:

```python
def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Process image for annotation display using ScalingManager with caching."""
    try:
        # Generate cache key
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        # Check cache first
        if image_hash in self._processed_cache:
            return self._processed_cache[image_hash], self._metadata_cache[image_hash]
        
        # Process image using existing ScalingManager
        processed_image, metadata = self.scaling_manager.process_image(
            image, 
            interpolation=InterpolationMethod.AREA
        )
        
        # Cache the results
        self._processed_cache[image_hash] = processed_image
        self._metadata_cache[image_hash] = metadata
        
        # Manage cache size for metadata
        if len(self._metadata_cache) > self.max_cache_size:
            # Remove oldest items
            oldest_key = next(iter(self._metadata_cache))
            del self._metadata_cache[oldest_key]
        
        return processed_image, metadata
```

### Prediction Cache

Implemented in both `SAM1Predictor` and `SAM2Predictor` classes, this cache stores segmentation results to avoid redundant computations:

```python
# Cache initialization in predictor classes
self.current_image_hash = None
self.prediction_cache = {}
self.max_cache_size = 50
```

Features:
- **Memory-Aware Caching**: Only caches predictions when memory usage is below warning threshold.
- **Current Image Preservation**: Option to retain cache entries for current image while clearing others.
- **Composite Keys**: Uses a combination of image hash and input parameters to identify cached entries.
- **Automatic Cleanup**: Clears older entries when cache size exceeds limit.

Implementation details:

```python
def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
    """Predict masks with memory management."""
    try:
        # Check memory status before prediction
        status_ok, message = self.memory_manager.check_memory_status()
        if not status_ok:
            raise RuntimeError(message)
            
        # Generate cache key and check cache
        cache_key = self._generate_cache_key(point_coords, point_labels, box)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Run prediction with optimizations
        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output
            )
            
            # Cache results if memory allows
            memory_info = self.memory_manager.get_gpu_memory_info()
            if memory_info['utilization'] < self.memory_manager.warning_threshold:
                self.prediction_cache[cache_key] = (masks, scores, logits)
                
                # Manage cache size
                if len(self.prediction_cache) > self.max_cache_size:
                    self.clear_cache(keep_current=True)
            
            return masks, scores, logits
    except Exception as e:
        # Try to recover memory
        self.memory_manager.optimize_memory(force=True)
        raise
```

## Memory Optimization Strategies

SAM Annotator employs several strategies to optimize memory usage:

1. **Periodic Memory Checks**: Regularly checks memory usage in the main application loop:

```python
# From annotator.py run() method
if hasattr(self.image_processor, 'get_memory_usage'):
    memory_usage = self.image_processor.get_memory_usage()
    if memory_usage > 1e9:  # More than 1GB
        self.logger.info("Clearing image cache due to high memory usage")
        self.image_processor.clear_cache()
                        
# Check GPU memory periodically
if hasattr(self.predictor, 'get_memory_usage'):
    gpu_memory = self.predictor.get_memory_usage()
    if gpu_memory > 0.8:  # Over 80% GPU memory
        self.predictor.optimize_memory()
```

2. **Threshold-Based Cache Management**: Clears caches when memory usage exceeds predefined thresholds.

3. **Explicit Garbage Collection**: Forces Python's garbage collector and PyTorch's CUDA cache clearing:

```python
def optimize_memory(self, force: bool = False) -> None:
    """Optimize GPU memory usage."""
    if self.device.type != 'cuda':
        return
        
    try:
        memory_info = self.get_gpu_memory_info()
        if force or memory_info['utilization'] >= self.warning_threshold:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Memory optimization performed")
    except Exception as e:
        self.logger.error(f"Error optimizing memory: {e}")
```

4. **TF32 Precision**: Enables TF32 precision on supported NVIDIA GPUs for better performance and memory efficiency:

```python
def _setup_gpu(self) -> None:
    """Configure GPU memory management."""
    try:
        if self.enable_memory_growth:
            # Enable memory growth
            for device in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, device)
        
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception as e:
        self.logger.error(f"Error setting up GPU memory management: {e}")
```

5. **Automatic Image Resizing**: Resizes large images to reduce memory footprint using the `ScalingManager`.

6. **WeakValueDictionary for Image Cache**: Uses weak references to allow the garbage collector to reclaim memory when needed.

## Real-Time Memory Monitoring

The SAM Annotator tool implements real-time memory monitoring that runs continuously in the main application loop. This proactive approach ensures that memory usage is kept in check during extended annotation sessions.

### Main Loop Monitoring

The main loop in both `SAMAnnotator` and derived classes continuously monitors memory usage:

```python
# Main event loop implementation from sam_annotator.py
while True:
    # Check image processor memory usage
    if hasattr(self.image_processor, 'get_memory_usage'):
        memory_usage = self.image_processor.get_memory_usage()
        if memory_usage > 1e9:  # More than 1GB
            self.logger.info("Clearing image cache")
            self.image_processor.clear_cache()
            
    # Check GPU memory
    if hasattr(self.predictor, 'get_memory_usage'):
        gpu_memory = self.predictor.get_memory_usage()
        if gpu_memory > 0.8:  # Over 80% GPU memory
            self.predictor.optimize_memory()
    
    # Rest of the event loop...
```

### Memory Monitoring Thresholds

The real-time monitoring system uses two types of thresholds:

1. **Absolute thresholds** for image cache (1GB)
2. **Percentage thresholds** for GPU memory (80%)

These values were determined based on extensive testing to provide the best balance between performance and stability.

### Adaptive Response

The monitoring system doesn't just detect high memory usage; it responds adaptively:

1. **For image processing cache**: When memory exceeds 1GB, it clears the image cache completely
2. **For GPU memory**: When usage exceeds 80%, it performs a targeted optimization that includes:
   - Clearing unused prediction caches
   - Running PyTorch's CUDA cache empty function
   - Triggering Python's garbage collector

### Automatic Recovery

If memory issues occur during predictions, the system attempts automatic recovery:

```python
try:
    # Prediction code...
except Exception as e:
    self.logger.error(f"Error in prediction: {str(e)}")
    # Try to recover memory
    self.memory_manager.optimize_memory(force=True)
    raise
```

This real-time monitoring and recovery system ensures that SAM Annotator remains stable even during long annotation sessions with large images or complex segmentation tasks.

## Troubleshooting Memory Issues

If you encounter memory issues when using SAM Annotator:

### Common Issues on Windows

1. **KeyError: 'formatted'**: This error occurs when the GPU memory information lacks the 'formatted' key. Now fixed with the `safe_get_memory_info()` method.
   
   Solution: Update to the latest version with the fix.

2. **Out of Memory Errors**: Windows systems may experience OOM errors with large images.
   
   Solution: Reduce the maximum image size in the configuration or use the CPU mode.

### Common Issues on Linux

1. **CUDA Out of Memory**: Linux systems might still have CUDA OOM errors with large batches.
   
   Solution: Set the environment variable `SAM_GPU_MEMORY_FRACTION=0.7` to limit GPU memory usage.

2. **nvidia-smi Not Found**: Some Linux distributions might lack proper NVIDIA driver setup.
   
   Solution: Install NVIDIA drivers or set `SAM_ENABLE_MEMORY_GROWTH=false` to use PyTorch's memory management.

### General Optimization Tips

1. **Environment Variables**:
   - `SAM_GPU_MEMORY_FRACTION`: Controls maximum GPU memory usage (default: 0.9)
   - `SAM_MEMORY_WARNING_THRESHOLD`: Sets memory warning level (default: 0.8)
   - `SAM_MEMORY_CRITICAL_THRESHOLD`: Sets memory critical level (default: 0.95)
   - `SAM_ENABLE_MEMORY_GROWTH`: Enables/disables memory growth (default: True)

2. **Reduce Image Size**: Configure smaller target image sizes to reduce memory footprint.

3. **Limit Batch Processing**: Process fewer images at once to reduce memory pressure.

## Testing Memory Management

SAM Annotator includes a test suite for memory management that validates:

1. **Memory allocation limits**: Tests allocation with different memory fractions.
2. **Memory growth behavior**: Validates behavior when memory limits are reached.
3. **Optimization effectiveness**: Measures memory recovery after optimization.

The test code includes a memory allocation test that tries to allocate large tensors while monitoring memory usage:

```python
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
            
            # If we hit critical threshold, break
            if not status_ok:
                logger.warning("Hit critical memory threshold!")
                break

        # Try to optimize memory
        logger.info("Attempting memory optimization...")
        memory_manager.optimize_memory(force=True)
        
        # Check memory after optimization
        post_opt_info = memory_manager.get_gpu_memory_info()
        logger.info(f"After optimization: {format_memory_info(post_opt_info)}")

    except Exception as e:
        logger.error(f"Error during memory test: {e}")
```

To run memory tests with different configurations, use:

```bash
TEST_MEMORY_FRACTIONS=0.9,0.7,0.5 python -m tests.test_memory_manager
```

---

This document covers the key aspects of memory management in SAM Annotator. For further details, refer to the code documentation and comments in the relevant files:
- `memory_manager.py`: Core memory management functionality
- `image_utils.py`: Image processing cache implementation
- `predictor.py`: Prediction caching systems 