"""
Rate limiter for API requests to prevent hitting rate limits.
"""
from typing import Dict, Optional, List, Deque
from threading import Lock
import time
import logging
from collections import deque
from datetime import datetime, timedelta

from irs_data_agent.utils.logging import get_logger
from irs_data_agent.utils.config import load_config

logger = get_logger(__name__)

class RateLimiter:
    """
    Rate limiter for API requests to prevent hitting rate limits.
    
    Implements configurable token bucket and sliding window algorithms
    to enforce rate limits across different time windows.
    """
    
    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        max_burst: Optional[int] = None
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_day: Maximum requests per day
            max_burst: Maximum burst size for token bucket
        """
        # Load configuration
        config = load_config()
        rate_limits = config.get("api", {}).get("rate_limits", {})
        
        # Set rate limits from config or defaults
        self.requests_per_minute = requests_per_minute or rate_limits.get("requests_per_minute", 60)
        self.requests_per_day = requests_per_day or rate_limits.get("requests_per_day", 5000)
        self.max_burst = max_burst or rate_limits.get("max_burst", self.requests_per_minute * 2)
        
        # Initialize token bucket for short-term rate limiting
        self.tokens = self.max_burst
        self.last_token_fill_time = time.time()
        self.token_rate = self.requests_per_minute / 60.0  # tokens per second
        
        # Initialize sliding window for long-term rate limiting
        self.request_times: Deque[float] = deque()
        
        # Thread safety
        self.lock = Lock()
        
        logger.info(f"Rate limiter initialized: {self.requests_per_minute}/minute, {self.requests_per_day}/day")
        
    def limit(self, block: bool = True) -> bool:
        """
        Check rate limits and delay if necessary.
        
        Args:
            block: Whether to block until request is allowed
            
        Returns:
            Whether the request is allowed
        """
        with self.lock:
            # Check daily limit
            if not self._check_daily_limit():
                if block:
                    self._wait_for_daily_limit()
                else:
                    logger.warning("Daily rate limit exceeded")
                    return False
            
            # Check and update token bucket
            current_time = time.time()
            time_passed = current_time - self.last_token_fill_time
            
            # Add new tokens based on time passed
            new_tokens = time_passed * self.token_rate
            self.tokens = min(self.max_burst, self.tokens + new_tokens)
            self.last_token_fill_time = current_time
            
            # Check if we have enough tokens
            if self.tokens < 1:
                if block:
                    # Calculate wait time and sleep
                    wait_time = (1 - self.tokens) / self.token_rate
                    logger.debug(f"Rate limit hit, waiting {wait_time:.2f} seconds")
                    
                    # Release lock while waiting
                    self.lock.release()
                    try:
                        time.sleep(wait_time)
                    finally:
                        # Ensure we reacquire the lock even if interrupted
                        self.lock.acquire()
                    
                    # Recalculate tokens after waiting
                    current_time = time.time()
                    time_passed = current_time - self.last_token_fill_time
                    new_tokens = time_passed * self.token_rate
                    self.tokens = min(self.max_burst, self.tokens + new_tokens)
                    self.last_token_fill_time = current_time
                else:
                    logger.warning("Minute rate limit exceeded")
                    return False
            
            # Consume a token
            self.tokens -= 1
            
            # Record request time
            self.request_times.append(time.time())
            
            # Clean up old request times
            self._clean_old_requests()
                
            return True
    
    def _check_daily_limit(self) -> bool:
        """
        Check if the daily limit has been exceeded.
        
        Returns:
            Whether the daily limit is ok
        """
        # Clean up old request times
        self._clean_old_requests()
            
        # Check if we've exceeded the daily limit
        return len(self.request_times) < self.requests_per_day
    
    def _clean_old_requests(self):
        """Remove request times older than 24 hours."""
        day_ago = time.time() - 86400  # 24 hours in seconds
        while self.request_times and self.request_times[0] < day_ago:
            self.request_times.popleft()
    
    def _wait_for_daily_limit(self):
        """Wait until daily limit allows a new request."""
        while self.request_times and len(self.request_times) >= self.requests_per_day:
            # Calculate time to wait
            oldest_request = self.request_times[0]
            wait_time = (oldest_request + 86400) - time.time()
            
            if wait_time > 0:
                logger.warning(f"Daily rate limit reached, waiting {wait_time:.2f} seconds")
                
                # Release lock while waiting
                self.lock.release()
                try:
                    time.sleep(min(wait_time, 60))  # Wait at most 60 seconds at a time
                finally:
                    # Ensure we reacquire the lock even if interrupted
                    self.lock.acquire()
                
                # Clean up old request times after waiting
                self._clean_old_requests()
            else:
                # If wait time is not positive, clean up old requests and check again
                self._clean_old_requests()
                if len(self.request_times) < self.requests_per_day:
                    break
    
    def get_status(self) -> Dict[str, float]:
        """
        Get the current status of the rate limiter.
        
        Returns:
            Status information
        """
        with self.lock:
            # Clean up old request times
            self._clean_old_requests()
                
            # Calculate statistics
            current_time = time.time()
            requests_last_minute = sum(1 for t in self.request_times if t > current_time - 60)
            requests_last_hour = sum(1 for t in self.request_times if t > current_time - 3600)
            requests_last_day = len(self.request_times)
            
            # Avoid division by zero
            minute_capacity_used = requests_last_minute / max(1, self.requests_per_minute)
            day_capacity_used = requests_last_day / max(1, self.requests_per_day)
            
            # Update token count before reporting
            time_passed = current_time - self.last_token_fill_time
            new_tokens = time_passed * self.token_rate
            current_tokens = min(self.max_burst, self.tokens + new_tokens)
            
            return {
                "tokens": current_tokens,
                "requests_last_minute": requests_last_minute,
                "requests_last_hour": requests_last_hour,
                "requests_last_day": requests_last_day,
                "requests_per_minute_limit": self.requests_per_minute,
                "requests_per_day_limit": self.requests_per_day,
                "minute_capacity_used": minute_capacity_used,
                "day_capacity_used": day_capacity_used
            }
            
    def reset(self):
        """Reset the rate limiter state."""
        with self.lock:
            self.tokens = self.max_burst
            self.last_token_fill_time = time.time()
            self.request_times.clear()
            
            logger.info("Rate limiter reset")
