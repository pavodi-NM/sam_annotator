# Memory Optimization & Lazy Loading Documentation

## Overview
This document outlines the optimization strategies implemented in the SAM Multi-Object Annotator tool to handle memory efficiently and improve performance through lazy loading and caching mechanisms.

## Table of Contents
1. [Lazy Loading](#lazy-loading)
2. [Memory Management](#memory-management)
3. [Caching Strategies](#caching-strategies)
4. [GPU Memory Optimization](#gpu-memory-optimization)
5. [Image Processing Optimization](#image-processing-optimization)

## Lazy Loading

### LazyImageLoader
The `LazyImageLoader` class implements lazy loading for images, only loading them into memory when actually needed.

```python
class LazyImageLoader:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self._image = None
        self._metadata = None
        self._lock = Lock()
```

Key features:
    - Loads images on-demand using property decorators
    - Caches metadata separately from actual image data
    - Thread-safe implementation using locks
    - Automatic memory cleanup when images are no longer needed

### Annotation Loading
Annotations are loaded lazily with the following optimizations:
    - Loads only when an image is accessed
    - Scales coordinates on-demand
    - Caches processed annotations for frequently accessed images

## Memory Management

### GPU Memory Manager
The `GPUMemoryManager` class handles GPU memory allocation and optimization:

```python
class GPUMemoryManager:
    def __init__(self, warning_threshold: float = 0.85, 
                 critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
```

Features:
    - Memory usage monitoring
    - Automatic cleanup when thresholds are exceeded
    - Warning system for high memory usage
    - Emergency memory recovery procedures

### Memory Optimization Strategies
1. **Automatic Cleanup**:
   - Clearing unused cache entries
   - Freeing GPU memory when unused
   - Garbage collection triggering

2. **Memory Thresholds**:
   - Warning threshold (85% usage)
   - Critical threshold (95% usage)
   - Automatic optimization triggers

## Caching Strategies

### Image Cache
    - Uses `WeakValueDictionary` for automatic cleanup
    - Caches processed images at display resolution
    - Maintains metadata separately from image data

### Prediction Cache
    - Caches SAM model predictions
    - Limited size cache with LRU eviction
    - Automatic invalidation on image change

### Cache Implementation:
```python
# Image cache with weak references
self.image_cache = WeakValueDictionary()

# Prediction cache with size limit
self.prediction_cache = {}
self.max_cache_size = 50
```

### Cache Optimization:
    - Size-based limits
    - Age-based eviction
    - Memory-aware caching
    - Selective cache clearing

## GPU Memory Optimization

### SAM Model Optimization
    - Batch processing optimization
    - TensorFlow 32 (TF32) enablement
    - CUDA optimization settings
    - Model weight management

### Memory Monitoring:
```python
def get_gpu_memory_info(self) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {'used': 0, 'total': 0, 'utilization': 0}
        
    gpu_memory = torch.cuda.memory_stats()
    allocated = gpu_memory.get('allocated_bytes.all.current', 0)
    reserved = gpu_memory.get('reserved_bytes.all.current', 0)
    total = torch.cuda.get_device_properties(0).total_memory
```

### Optimization Triggers:
    - Memory usage thresholds
    - Performance monitoring
    - Error recovery
    - Automatic optimization

## Image Processing Optimization

### Display Size Management
    - Automatic resizing for display
    - Aspect ratio preservation
    - Resolution optimization

### Scaling Operations:
```python
def scale_to_display(self, coords: Union[np.ndarray, List[Tuple[int, int]], 
                    Tuple[int, int]], coord_type: str = 'point') -> np.ndarray:
    if isinstance(coords, tuple):
        coords = np.array([coords])
    return self.scaling_manager.to_display_space(coords, coord_type)
```

### Performance Optimizations:
    - Cached scaling operations
    - Efficient coordinate transformations
    - Optimized mask generation

## Best Practices

### Memory Management
    1. Monitor GPU memory usage regularly
    2. Clear caches when switching tasks
    3. Implement proper cleanup in error handlers

### Cache Usage
    1. Set appropriate cache size limits
    2. Use weak references when possible
    3. Implement cache invalidation strategies

### Image Processing
    1. Process images at appropriate resolution
    2. Cache processed results
    3. Clean up unused resources

## Error Handling and Recovery

### Memory-Related Errors
```python
try:
    # Memory-intensive operation
    status_ok, message = self.memory_manager.check_memory_status()
    if not status_ok:
        self.memory_manager.optimize_memory(force=True)
except Exception as e:
    self.logger.error(f"Memory error: {str(e)}")
    self.clear_cache()  # Emergency cleanup
```

### Recovery Strategies
    1. Automatic cache clearing
    2. Forced memory optimization
    3. Graceful degradation options

## Configuration

### Memory Settings
    - Warning threshold: 85% GPU memory usage
    - Critical threshold: 95% GPU memory usage
    - Cache size limits: 50 entries default

### Cache Settings
    - Maximum cache size
    - Cache lifetime
    - Cleanup triggers

## Monitoring and Debugging

### Logging
    - Memory usage logging
    - Cache hit/miss rates
    - Performance metrics

### Debugging Tools
    - Memory usage tracking
    - Cache inspection
    - Performance profiling

## Future Improvements

### Planned Optimizations
    1. Dynamic cache sizing based on available memory
    2. More sophisticated prediction caching
    3. Advanced memory prediction algorithms
    4. Multi-GPU support optimization

### Performance Enhancements
    1. Improved batch processing
    2. Better memory prediction
    3. More efficient cache management
    4. Enhanced error recovery

## Conclusion
    These optimization strategies enable efficient handling of large datasets while maintaining responsive performance. Regular monitoring and adjustment of these settings may be necessary based on specific use cases and hardware configurations.