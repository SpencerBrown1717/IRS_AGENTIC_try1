"""
Parallel processing utilities for IRS data agent.
"""
from typing import List, Callable, Any, TypeVar, Generic, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import traceback

from irs_data_agent.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')

def parallel_process(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: int = 4,
    use_processes: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = False,
    chunk_size: int = 1
) -> List[R]:
    """
    Process items in parallel using either threads or processes.
    
    Args:
        items: List of items to process
        process_func: Function to process each item
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        timeout: Optional timeout per item
        show_progress: Whether to show progress in logs
        chunk_size: Number of items to process in each worker task
        
    Returns:
        List of results in the same order as input items
    """
    if not items:
        return []
    
    # Limit max_workers to a reasonable number
    max_workers = min(max_workers, len(items), 32)
    
    # Choose executor based on whether we want threads or processes
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    results = [None] * len(items)
    errors = []
    
    # Group items into chunks
    if chunk_size > 1:
        chunked_items = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i+chunk_size]
            chunked_items.append((i, chunk))
        
        # Define a function to process a chunk
        def process_chunk(chunk_data):
            start_idx, chunk_items = chunk_data
            chunk_results = []
            chunk_errors = []
            
            for i, item in enumerate(chunk_items):
                try:
                    result = process_func(item)
                    chunk_results.append((start_idx + i, result))
                except Exception as e:
                    chunk_errors.append((start_idx + i, e))
                    # Log the error with traceback
                    logger.error(f"Error processing item {start_idx + i}: {str(e)}")
                    logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            return chunk_results, chunk_errors
        
        # Process chunks in parallel
        with executor_class(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunked_items}
            
            completed = 0
            start_time = time.time()
            
            for future in as_completed(futures):
                chunk = futures[future]
                
                try:
                    chunk_results, chunk_errors = future.result(timeout=timeout)
                    
                    # Store results
                    for idx, result in chunk_results:
                        results[idx] = result
                    
                    # Store errors
                    errors.extend(chunk_errors)
                    
                except Exception as e:
                    start_idx, chunk_items = chunk
                    for i in range(len(chunk_items)):
                        errors.append((start_idx + i, e))
                    logger.error(f"Error processing chunk starting at index {start_idx}: {str(e)}")
                    logger.debug(f"Error traceback: {traceback.format_exc()}")
                
                completed += 1
                
                if show_progress:
                    elapsed = time.time() - start_time
                    percent = (completed / len(chunked_items)) * 100
                    logger.info(f"Progress: {completed}/{len(chunked_items)} chunks ({percent:.1f}%), Elapsed: {elapsed:.1f}s")
        
    else:
        # Process individual items in parallel
        with executor_class(max_workers=max_workers) as executor:
            futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}
            
            completed = 0
            start_time = time.time()
            
            for future in as_completed(futures):
                idx = futures[future]
                
                try:
                    results[idx] = future.result(timeout=timeout)
                except Exception as e:
                    errors.append((idx, e))
                    logger.error(f"Error processing item {idx}: {str(e)}")
                    logger.debug(f"Error traceback: {traceback.format_exc()}")
                
                completed += 1
                
                if show_progress and completed % max(1, len(items) // 10) == 0:
                    elapsed = time.time() - start_time
                    percent = (completed / len(items)) * 100
                    logger.info(f"Progress: {completed}/{len(items)} items ({percent:.1f}%), Elapsed: {elapsed:.1f}s")
    
    # Log errors
    if errors:
        logger.warning(f"Encountered {len(errors)} errors during parallel processing")
        
        # Set error items to None
        for idx, _ in errors:
            results[idx] = None
    
    return results

def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    max_workers: int = 4,
    use_processes: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = False
) -> List[R]:
    """
    Simplified version of parallel_process for direct mapping.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        timeout: Optional timeout per item
        show_progress: Whether to show progress in logs
        
    Returns:
        List of results
    """
    return parallel_process(
        items=items,
        process_func=func,
        max_workers=max_workers,
        use_processes=use_processes,
        timeout=timeout,
        show_progress=show_progress
    )

def execute_with_retries(
    func: Callable[..., R],
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_exceptions: List[type] = None,
    **kwargs
) -> R:
    """
    Execute a function with retries and exponential backoff.
    
    Args:
        func: Function to execute
        *args: Positional arguments to pass to func
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay for each retry
        retry_exceptions: List of exception types to retry on. If None, retry on all exceptions.
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Function result
        
    Raises:
        Exception: The last exception encountered if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            # Check if we should retry this exception type
            if retry_exceptions is not None and not any(isinstance(e, exc_type) for exc_type in retry_exceptions):
                logger.info(f"Not retrying exception of type {type(e).__name__} as it's not in the retry_exceptions list")
                raise
            
            if attempt < max_retries:
                delay = retry_delay * (backoff_factor ** attempt)
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. Retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    if last_exception:
        raise last_exception
    
    # This should never happen, but just in case
    raise RuntimeError("Unknown error in execute_with_retries")
