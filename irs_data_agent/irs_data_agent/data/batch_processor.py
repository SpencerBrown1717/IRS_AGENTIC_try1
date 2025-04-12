"""
Batch processing for IRS data queries.
"""
from typing import Dict, List, Any, Optional, Callable, Union
import os
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

from irs_data_agent.core.workflow import Workflow
from irs_data_agent.core.state import State
from irs_data_agent.agents.planning_agent import PlanningAgent
from irs_data_agent.agents.retrieval_agent import RetrievalAgent
from irs_data_agent.agents.processing_agent import ProcessingAgent
from irs_data_agent.agents.error_agent import ErrorAgent
from irs_data_agent.api.irs_client import IRSClient
from irs_data_agent.data.cache_manager import CacheManager
from irs_data_agent.utils.logging import get_logger

logger = get_logger(__name__)

class BatchProcessor:
    """
    Batch processor for running multiple IRS data queries in parallel.
    """
    
    def __init__(
        self,
        queries: List[str],
        year: Optional[int] = None,
        output_format: str = "json",
        save_dir: Union[str, Path] = "./results",
        workers: int = 2,
        use_cache: bool = True,
        max_retries: int = 2
    ):
        """
        Initialize the batch processor.
        
        Args:
            queries: List of search queries to process
            year: Optional tax year filter for all queries
            output_format: Output format (text, json, csv)
            save_dir: Directory to save results
            workers: Number of parallel workers
            use_cache: Whether to use cache
            max_retries: Maximum number of retries per query
        """
        self.queries = queries
        self.year = year
        self.output_format = output_format
        self.save_dir = Path(save_dir)
        self.workers = max(1, min(workers, 8))  # Limit between 1 and 8
        self.use_cache = use_cache
        self.max_retries = max_retries
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create a unique batch ID
        self.batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Initialize client and agents
        self.client = IRSClient()
        
        # Initialize cache if enabled
        self.cache = CacheManager() if use_cache else None
        
        logger.info(f"Initialized batch processor with {len(queries)} queries and {workers} workers")
    
    def process(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all queries in the batch.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results for all queries
        """
        logger.info(f"Starting batch processing of {len(self.queries)} queries")
        
        # Create a metadata file for the batch
        self._create_batch_metadata()
        
        # Track batch statistics
        start_time = time.time()
        completed = 0
        results = []
        
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_query = {
                executor.submit(self._process_query, query, idx): (query, idx)
                for idx, query in enumerate(self.queries)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_query):
                query, idx = future_to_query[future]
                
                try:
                    result = future.result()
                    completed += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(completed, len(self.queries), query)
                        
                    # Save result
                    self._save_result(result, query)
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {str(e)}", exc_info=True)
                    
                    # Add error result
                    error_result = {
                        "query": query,
                        "success": False,
                        "error": str(e),
                        "index": idx
                    }
                    
                    self._save_result(error_result, query, is_error=True)
                    results.append(error_result)
                    
                    completed += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(completed, len(self.queries), query)
        
        # Update batch metadata with completion info
        self._update_batch_metadata(results, time.time() - start_time)
        
        logger.info(f"Batch processing completed: {completed}/{len(self.queries)} queries processed")
        return results
    
    def _process_query(self, query: str, index: int) -> Dict[str, Any]:
        """
        Process a single query.
        
        Args:
            query: Query to process
            index: Query index in the batch
            
        Returns:
            Query result
        """
        logger.info(f"Processing query {index+1}/{len(self.queries)}: {query}")
        
        # Create state
        state = State(
            query=query,
            year=self.year,
            is_batch_mode=True,
            batch_id=self.batch_id
        )
        
        # Initialize workflow components
        planning_agent = PlanningAgent()
        retrieval_agent = RetrievalAgent(client=self.client)
        processing_agent = ProcessingAgent()
        error_agent = ErrorAgent()
        
        # Create workflow
        workflow = Workflow(
            state=state,
            planning_agent=planning_agent,
            retrieval_agent=retrieval_agent,
            processing_agent=processing_agent,
            error_agent=error_agent,
            cache=self.cache
        )
        
        # Run workflow
        start_time = time.time()
        retries = 0
        success = False
        error = None
        
        while retries <= self.max_retries and not success:
            try:
                result = workflow.run()
                success = True
            except Exception as e:
                logger.warning(f"Error on attempt {retries+1}/{self.max_retries+1} for query '{query}': {str(e)}")
                error = str(e)
                retries += 1
                
                if retries <= self.max_retries:
                    # Reset state and retry
                    state = error_agent.reset_state_after_error(state)
                    
                    # Create new workflow with reset state
                    workflow = Workflow(
                        state=state,
                        planning_agent=planning_agent,
                        retrieval_agent=retrieval_agent,
                        processing_agent=processing_agent,
                        error_agent=error_agent,
                        cache=self.cache
                    )
                    
                    # Wait before retrying
                    time.sleep(2 ** retries)
        
        processing_time = time.time() - start_time
        
        if success:
            # Add batch metadata
            result["batch_id"] = self.batch_id
            result["index"] = index
            result["processing_time"] = processing_time
            result["success"] = True
            return result
        else:
            # Return error result
            return {
                "query": query,
                "batch_id": self.batch_id,
                "index": index,
                "processing_time": processing_time,
                "success": False,
                "error": error,
                "results": []
            }
    
    def _save_result(self, result: Dict[str, Any], query: str, is_error: bool = False):
        """
        Save a query result to file.
        
        Args:
            result: Query result
            query: The query string
            is_error: Whether this is an error result
        """
        # Create a safe filename from query
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:50])
        index = result.get("index", 0)
        
        if is_error:
            filename = f"{index:04d}_error_{safe_query}.json"
        else:
            filename = f"{index:04d}_{safe_query}.json"
            
        file_path = self.save_dir / filename
        
        # Save result
        try:
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
                
            logger.debug(f"Saved result to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving result to {file_path}: {str(e)}")
    
    def _create_batch_metadata(self):
        """Create batch metadata file."""
        metadata = {
            "batch_id": self.batch_id,
            "created_at": datetime.now().isoformat(),
            "query_count": len(self.queries),
            "year": self.year,
            "output_format": self.output_format,
            "workers": self.workers,
            "use_cache": self.use_cache,
            "status": "running",
            "queries": self.queries[:10]  # Include only first 10 queries
        }
        
        if len(self.queries) > 10:
            metadata["queries_truncated"] = True
            metadata["total_queries"] = len(self.queries)
            
        # Save metadata
        file_path = self.save_dir / f"{self.batch_id}_metadata.json"
        try:
            with open(file_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"Created batch metadata file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error creating batch metadata file: {str(e)}")
    
    def _update_batch_metadata(self, results: List[Dict[str, Any]], elapsed_time: float):
        """
        Update batch metadata with completion information.
        
        Args:
            results: All query results
            elapsed_time: Total elapsed time for batch
        """
        # Count successes and failures
        successes = sum(1 for r in results if r.get("success", False))
        failures = len(results) - successes
        
        # Load existing metadata
        file_path = self.save_dir / f"{self.batch_id}_metadata.json"
        try:
            with open(file_path, "r") as f:
                metadata = json.load(f)
                
            # Update metadata
            metadata["completed_at"] = datetime.now().isoformat()
            metadata["elapsed_time"] = elapsed_time
            metadata["status"] = "completed"
            metadata["successes"] = successes
            metadata["failures"] = failures
            
            # Calculate statistics
            if results:
                result_counts = [len(r.get("results", [])) for r in results if r.get("success", False)]
                if result_counts:
                    metadata["avg_results_per_query"] = sum(result_counts) / len(result_counts)
                    metadata["max_results"] = max(result_counts) if result_counts else 0
                    metadata["min_results"] = min(result_counts) if result_counts else 0
                    metadata["total_results"] = sum(result_counts)
            
            # Save updated metadata
            with open(file_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"Updated batch metadata file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error updating batch metadata file: {str(e)}")
            
    def get_status(self) -> Dict[str, Any]:
        """
        Get batch processing status.
        
        Returns:
            Status information
        """
        # Try to load metadata
        file_path = self.save_dir / f"{self.batch_id}_metadata.json"
        try:
            with open(file_path, "r") as f:
                metadata = json.load(f)
                
            # Count result files
            result_count = sum(1 for f in os.listdir(self.save_dir) 
                             if f.startswith(f"{self.batch_id}") and f.endswith(".json"))
            
            return {
                "batch_id": self.batch_id,
                "status": metadata.get("status", "unknown"),
                "queries": len(self.queries),
                "completed": result_count,
                "created_at": metadata.get("created_at"),
                "completed_at": metadata.get("completed_at")
            }
            
        except Exception as e:
            logger.error(f"Error getting batch status: {str(e)}")
            
            return {
                "batch_id": self.batch_id,
                "status": "error",
                "queries": len(self.queries),
                "error": str(e)
            }
