"""Utility decorators for the Mean Reversion Trading System."""

import time
import functools
from typing import Any, Callable, Optional, Type, Union
from .logger import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Exception types to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {str(e)}"
                        )
                        raise e
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_attempts}. "
                        f"Retrying in {current_delay:.1f}s. Error: {str(e)}"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def timing(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to measure
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Function {func.__name__} executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.4f}s. Error: {str(e)}"
            )
            raise
    
    return wrapper


def cache_result(ttl: Optional[float] = None):
    """
    Decorator to cache function results with optional TTL.
    
    Args:
        ttl: Time to live in seconds. If None, cache never expires.
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if result is cached and not expired
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (current_time - timestamp) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    # Remove expired entry
                    del cache[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        # Add cache management methods
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'keys': list(cache.keys())
        }
        
        return wrapper
    return decorator


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Keyword arguments mapping parameter names to validation functions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}' "
                            f"with value {value} in function {func.__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
