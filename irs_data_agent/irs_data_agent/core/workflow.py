"""
Enhanced workflow engine for orchestrating IRS data retrieval and processing.
"""
from typing import Dict, List, Any, Optional, Callable
from irs_data_agent.core.state import State
from irs_data_agent.agents.planning_agent import PlanningAgent
from irs_data_agent.agents.retrieval_agent import RetrievalAgent
from irs_data_agent.agents.processing_agent import ProcessingAgent
from irs_data_agent.agents.error_agent import ErrorAgent
from irs_data_agent.data.cache_manager import CacheManager
from irs_data_agent.utils.logging import get_logger

logger = get_logger(__name__)

class Workflow:
    """
    Enhanced workflow engine that orchestrates the planning, retrieval, and processing
    of IRS data using a multi-agent approach.
    """
    
    def __init__(
        self,
        state: State,
        planning_agent: PlanningAgent,
        retrieval_agent: RetrievalAgent,
        processing_agent: ProcessingAgent,
        error_agent: ErrorAgent,
        cache: Optional[CacheManager] = None,
        max_iterations: int = 5,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Initialize the workflow with state and agents.
        
        Args:
            state: Current workflow state
            planning_agent: Agent for planning actions
            retrieval_agent: Agent for retrieving data
            processing_agent: Agent for processing data
            error_agent: Agent for handling errors
            cache: Optional cache manager
            max_iterations: Maximum number of planning-execution cycles
            progress_callback: Optional callback for progress updates
        """
        self.state = state
        self.planning_agent = planning_agent
        self.retrieval_agent = retrieval_agent
        self.processing_agent = processing_agent
        self.error_agent = error_agent
        self.cache = cache
        self.max_iterations = max_iterations
        self.progress_callback = progress_callback
        
    def run(self) -> Dict[str, Any]:
        """
        Run the workflow to completion.
        
        Returns:
            Dict containing results and metadata
        """
        logger.info(f"Starting workflow with query: {self.state.query}")
        
        # Set state to running
        self.state.set_status("running")
        
        # Check cache first if enabled
        if self.cache and not self.state.is_batch_mode:
            cache_key = self._generate_cache_key()
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                logger.info(f"Found cached result for query: {self.state.query}")
                
                # Update state with cached results
                if 'results' in cached_result:
                    self.state.update_with_results(cached_result['results'])
                
                self.state.set_status("completed")
                self.state.metadata["source"] = "cache"
                
                return {
                    "query": self.state.query,
                    "results": self.state.results,
                    "complete": True,
                    "source": "cache",
                    "metadata": self.state.metadata
                }
        
        try:
            # Main workflow loop
            for iteration in range(self.max_iterations):
                logger.info(f"Workflow iteration {iteration + 1}/{self.max_iterations}")
                
                # Update progress if callback provided
                if self.progress_callback:
                    progress_message = f"Planning iteration {iteration + 1}/{self.max_iterations}"
                    self.progress_callback(iteration, self.max_iterations, progress_message)
                
                # 1. Planning Phase
                if not self._run_planning_phase():
                    break
                
                # 2. Retrieval Phase
                if not self._run_retrieval_phase():
                    break
                
                # 3. Processing Phase
                if not self._run_processing_phase():
                    break
                
                # Check if we're done
                if self.state.is_complete():
                    logger.info("Workflow completed successfully")
                    break
            
            # Cache results if enabled and successful
            if (self.cache and 
                not self.state.is_batch_mode and 
                self.state.is_successful()):
                
                cache_key = self._generate_cache_key()
                cache_data = {
                    'query': self.state.query,
                    'results': self.state.results,
                    'metadata': self.state.metadata
                }
                self.cache.set(cache_key, cache_data)
                logger.info(f"Cached results for query: {self.state.query}")
                
            # Prepare final result
            return self._prepare_result()
            
        except Exception as e:
            logger.error(f"Error during workflow execution: {str(e)}", exc_info=True)
            
            # Handle error with error agent
            error_handled = self._handle_error(e)
            
            # Update state status
            if not error_handled:
                self.state.set_status("failed", error=e)
                
            return self._prepare_result()
        
    def _run_planning_phase(self) -> bool:
        """
        Run the planning phase of the workflow.
        
        Returns:
            Boolean indicating success
        """
        try:
            logger.info("Starting planning phase")
            
            # Generate plan
            plan = self.planning_agent.create_plan(self.state)
            self.state.current_plan = plan
            
            logger.info(f"Created plan with {len(plan.steps)} steps")
            return True
            
        except Exception as e:
            logger.error(f"Error in planning phase: {str(e)}", exc_info=True)
            self._handle_error(e, phase="planning")
            return False
    
    def _run_retrieval_phase(self) -> bool:
        """
        Run the retrieval phase of the workflow.
        
        Returns:
            Boolean indicating success
        """
        try:
            logger.info("Starting retrieval phase")
            
            # Execute retrieval steps from the plan
            retrieval_results = self.retrieval_agent.execute_retrieval(self.state)
            
            # Update state with retrieval results
            self.state.retrieved_data = retrieval_results
            
            logger.info(f"Completed retrieval phase with {len(retrieval_results)} results")
            return True
            
        except Exception as e:
            logger.error(f"Error in retrieval phase: {str(e)}", exc_info=True)
            self._handle_error(e, phase="retrieval")
            return False
    
    def _run_processing_phase(self) -> bool:
        """
        Run the processing phase of the workflow.
        
        Returns:
            Boolean indicating success
        """
        try:
            logger.info("Starting processing phase")
            
            # Process the retrieved data
            processing_results = self.processing_agent.process_data(self.state)
            
            # Update state with processing results
            self.state.processed_data = processing_results
            self.state.update_with_results(processing_results.get("results", []))
            
            # Mark workflow as complete if we have results
            if self.state.results:
                self.state.set_status("completed")
            
            logger.info(f"Completed processing phase with {len(self.state.results)} results")
            return True
            
        except Exception as e:
            logger.error(f"Error in processing phase: {str(e)}", exc_info=True)
            self._handle_error(e, phase="processing")
            return False
            
    def _handle_error(self, error: Exception, phase: str = "general") -> bool:
        """
        Handle an error during workflow execution.
        
        Args:
            error: The error that occurred
            phase: The phase where the error occurred
            
        Returns:
            Boolean indicating if the error was handled
        """
        logger.info(f"Handling error in {phase} phase: {str(error)}")
        
        try:
            # Let the error agent handle the error
            handled, recovery_action = self.error_agent.handle_error(
                error=error,
                state=self.state,
                phase=phase
            )
            
            if handled:
                logger.info(f"Error in {phase} phase handled successfully")
                
                # Execute recovery action if provided
                if recovery_action:
                    logger.info(f"Executing recovery action: {recovery_action}")
                    # Implementation would depend on the type of recovery actions supported
                
                return True
            else:
                logger.warning(f"Error in {phase} phase could not be handled")
                self.state.set_status("failed", error=error)
                return False
                
        except Exception as e:
            logger.error(f"Error while handling error: {str(e)}", exc_info=True)
            self.state.set_status("failed", error=error)
            return False
            
    def _prepare_result(self) -> Dict[str, Any]:
        """
        Prepare the final result dictionary.
        
        Returns:
            Result dictionary
        """
        result = {
            "query": self.state.query,
            "results": self.state.results,
            "status": self.state.status,
            "complete": self.state.is_complete(),
            "successful": self.state.is_successful(),
            "total_results": len(self.state.results),
            "metadata": self.state.metadata
        }
        
        # Add error information if present
        if self.state.error:
            result["error"] = self.state.error
            
        # Add execution stats
        if self.state.execution_stats:
            result["stats"] = self.state.execution_stats
            
        return result
    
    def _generate_cache_key(self) -> str:
        """
        Generate a cache key for the current state.
        
        Returns:
            Cache key string
        """
        # Create a key based on query parameters
        key_parts = [self.state.query]
        
        if self.state.year:
            key_parts.append(f"year:{self.state.year}")
            
        if self.state.form:
            key_parts.append(f"form:{self.state.form}")
            
        return "|".join(key_parts)
    
    def reset(self):
        """Reset the workflow state."""
        self.state.reset()
        logger.info("Workflow state has been reset")
