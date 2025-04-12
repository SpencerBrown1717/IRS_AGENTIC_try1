"""
Error handling agent for IRS data operations.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
from irs_data_agent.core.state import State
from irs_data_agent.utils.logging import get_logger

import time
import traceback
import json
import os
import sys
from datetime import datetime

logger = get_logger(__name__)

class ErrorAgent:
    """
    Agent responsible for handling errors and implementing recovery strategies.
    """
    
    def __init__(
        self,
        max_retry_attempts: int = 3,
        report_errors: bool = True,
        error_log_path: str = "logs/errors.log"
    ):
        """
        Initialize the error agent.
        
        Args:
            max_retry_attempts: Maximum number of retry attempts
            report_errors: Whether to report errors to log file
            error_log_path: Path to error log file
        """
        self.max_retry_attempts = max_retry_attempts
        self.report_errors = report_errors
        self.error_log_path = error_log_path
        
        # Create logs directory if it doesn't exist
        if report_errors and not os.path.exists(os.path.dirname(error_log_path)):
            os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
            
    def handle_error(
        self, 
        error: Union[Exception, str, Dict[str, Any]],
        state: State,
        phase: str = "general"
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Handle an error during workflow execution.
        
        Args:
            error: The error that occurred
            state: Current workflow state
            phase: The phase where the error occurred
            
        Returns:
            Tuple of (success, recovery_action) where success is a boolean indicating
            if the error was handled successfully, and recovery_action is an optional
            dictionary with recovery action details
        """
        logger.info(f"Handling error in {phase} phase")
        
        # Normalize error to dict format
        error_dict = self._normalize_error(error)
        
        # Log error
        self._log_error(error_dict, state, phase)
        
        # Check if error is recoverable
        is_recoverable, recovery_strategy = self._is_recoverable(error_dict, phase)
        
        if not is_recoverable:
            logger.warning(f"Error in {phase} phase is not recoverable")
            return False, None
            
        # Implement recovery strategy
        recovery_action = self._implement_recovery(recovery_strategy, state, error_dict)
        
        if recovery_action:
            logger.info(f"Recovery action created: {recovery_action['type']}")
            return True, recovery_action
        else:
            logger.warning(f"Failed to create recovery action for {phase} phase error")
            return False, None
            
    def _normalize_error(self, error: Union[Exception, str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize error to dictionary format.
        
        Args:
            error: Error in various formats
            
        Returns:
            Error dictionary
        """
        if isinstance(error, dict):
            # Already a dictionary, ensure it has the required keys
            error_dict = error.copy()
            if "type" not in error_dict:
                error_dict["type"] = "unknown"
            if "message" not in error_dict:
                error_dict["message"] = str(error_dict)
            return error_dict
            
        elif isinstance(error, Exception):
            # Extract information from exception
            error_dict = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
            
            # Add more details for specific exception types
            if hasattr(error, "response"):
                # Handle requests.exceptions.HTTPError
                response = getattr(error, "response", None)
                if response:
                    error_dict["status_code"] = getattr(response, "status_code", None)
                    error_dict["url"] = getattr(response, "url", None)
                    try:
                        error_dict["response_body"] = response.text[:500]  # Truncate to avoid huge logs
                    except:
                        pass
                        
            return error_dict
            
        else:
            # Convert string or other type to dictionary
            return {
                "type": "string_error" if isinstance(error, str) else type(error).__name__,
                "message": str(error)
            }
            
    def _log_error(self, error: Dict[str, Any], state: State, phase: str):
        """
        Log error to file and state.
        
        Args:
            error: Error dictionary
            state: Current workflow state
            phase: Phase where the error occurred
        """
        # Add error to state
        if "error" not in state.metadata:
            state.metadata["error"] = []
            
        error_entry = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "error_type": error.get("type", "unknown"),
            "error_message": error.get("message", ""),
        }
        
        state.metadata["error"].append(error_entry)
        
        # Log to console
        logger.error(f"Error in {phase} phase: {error.get('type')}: {error.get('message')}")
        
        # Log to file if enabled
        if self.report_errors:
            try:
                # Create error log entry
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "phase": phase,
                    "query": state.query,
                    "error": error,
                    "state_info": {
                        "has_results": len(state.results) > 0,
                        "has_plan": state.current_plan is not None,
                        "form": state.form,
                        "year": state.year
                    }
                }
                
                # Append to error log file
                with open(self.error_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                    
            except Exception as e:
                logger.error(f"Failed to write to error log: {str(e)}")
                
    def _is_recoverable(self, error: Dict[str, Any], phase: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if an error is recoverable and what strategy to use.
        
        Args:
            error: Error dictionary
            phase: Phase where the error occurred
            
        Returns:
            Tuple of (is_recoverable, recovery_strategy)
        """
        error_type = error.get("type", "").lower()
        error_message = error.get("message", "").lower()
        status_code = error.get("status_code")
        
        # Handle timeout errors
        if ("timeout" in error_type or 
            "timeout" in error_message or 
            "timed out" in error_message):
            return True, "retry_with_backoff"
            
        # Handle rate limit errors
        if ((status_code == 429) or 
            ("rate" in error_message and "limit" in error_message) or
            ("too many requests" in error_message)):
            return True, "wait_and_retry"
            
        # Handle connection errors
        if ("connection" in error_type or 
            "connectionerror" in error_type or
            "connectexception" in error_type):
            return True, "retry_with_backoff"
            
        # Handle not found errors
        if (status_code == 404 or 
            "not found" in error_message or
            "notfound" in error_type):
            return True, "broaden_search" if phase == "retrieval" else "skip_and_continue"
            
        # Handle bad request errors
        if (status_code == 400 or
            "bad request" in error_message or
            "badrequest" in error_type or
            "invalid" in error_message):
            return True, "simplify_request"
            
        # Handle other HTTP errors
        if (status_code and status_code >= 500) or "server error" in error_message:
            return True, "retry_with_backoff"
            
        # Handle authentication errors
        if (status_code == 401 or status_code == 403 or
            "unauthorized" in error_message or 
            "forbidden" in error_message or
            "authentication" in error_message):
            return False, None  # Authentication errors aren't recoverable automatically
            
        # Handle value errors and type errors
        if ("valueerror" in error_type or 
            "typeerror" in error_type or
            "keyerror" in error_type):
            return True, "fix_data_format" if phase == "processing" else "skip_and_continue"
            
        # Default handling based on phase
        if phase == "planning":
            return True, "simplified_plan"
        elif phase == "retrieval":
            return True, "alternate_retrieval"
        elif phase == "processing":
            return True, "skip_processing"
        else:
            return False, None
            
    def _implement_recovery(
        self, 
        strategy: str, 
        state: State,
        error: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Implement a recovery strategy.
        
        Args:
            strategy: Recovery strategy
            state: Current workflow state
            error: Error dictionary
            
        Returns:
            Recovery action dictionary or None if recovery not possible
        """
        if not strategy:
            return None
            
        # Track retry attempts
        if "retry_count" not in state.metadata:
            state.metadata["retry_count"] = {}
            
        if strategy not in state.metadata["retry_count"]:
            state.metadata["retry_count"][strategy] = 0
            
        # Check if max retries exceeded
        if state.metadata["retry_count"][strategy] >= self.max_retry_attempts:
            logger.warning(f"Max retry attempts ({self.max_retry_attempts}) exceeded for strategy: {strategy}")
            return None
            
        # Increment retry count
        state.metadata["retry_count"][strategy] += 1
        
        # Implement specific strategies
        if strategy == "retry_with_backoff":
            backoff_seconds = 2 ** state.metadata["retry_count"][strategy]
            return {
                "type": "retry",
                "backoff_seconds": backoff_seconds,
                "max_retries": self.max_retry_attempts,
                "current_retry": state.metadata["retry_count"][strategy]
            }
            
        elif strategy == "wait_and_retry":
            wait_seconds = 60  # Default wait for rate limiting
            return {
                "type": "wait_and_retry",
                "wait_seconds": wait_seconds,
                "max_retries": self.max_retry_attempts,
                "current_retry": state.metadata["retry_count"][strategy]
            }
            
        elif strategy == "broaden_search":
            return {
                "type": "broaden_search",
                "original_query": state.query,
                "simplify": True
            }
            
        elif strategy == "simplify_request":
            return {
                "type": "simplify_request",
                "original_params": error.get("params", {})
            }
            
        elif strategy == "skip_and_continue":
            return {
                "type": "skip",
                "continue_workflow": True
            }
            
        elif strategy == "simplified_plan":
            return {
                "type": "create_simple_plan",
                "max_steps": 3
            }
            
        elif strategy == "alternate_retrieval":
            return {
                "type": "alternate_retrieval",
                "use_alternative_source": True
            }
            
        elif strategy == "skip_processing":
            return {
                "type": "skip_processing",
                "use_raw_data": True
            }
            
        elif strategy == "fix_data_format":
            return {
                "type": "fix_data_format",
                "sanitize_input": True
            }
            
        else:
            logger.warning(f"Unknown recovery strategy: {strategy}")
            return None
            
    def reset_state_after_error(self, state: State) -> State:
        """
        Reset state to a clean state after an unrecoverable error.
        
        Args:
            state: Current workflow state
            
        Returns:
            Reset state
        """
        # Create a copy of the state with original query parameters
        new_state = State(
            query=state.query,
            year=state.year,
            form=state.form,
            is_batch_mode=state.is_batch_mode,
            batch_id=state.batch_id
        )
        
        # Copy any error information
        if "error" in state.metadata:
            new_state.metadata["previous_errors"] = state.metadata["error"]
            
        return new_state
