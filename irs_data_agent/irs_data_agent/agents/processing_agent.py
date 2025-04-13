"""
Processing agent for IRS data processing operations.
"""
from typing import Dict, List, Any, Optional, Callable
from irs_data_agent.core.state import State
from irs_data_agent.utils.logging import get_logger
from irs_data_agent.utils.parallel import parallel_process

import time
from collections import defaultdict
from datetime import datetime

logger = get_logger(__name__)

class ProcessingAgent:
    """
    Agent responsible for processing retrieved IRS data.
    """
    
    def __init__(
        self,
        max_items_per_batch: int = 100,
        parallelize: bool = True,
        num_workers: int = 4
    ):
        """
        Initialize the processing agent.
        
        Args:
            max_items_per_batch: Maximum items to process in a batch
            parallelize: Whether to use parallel processing
            num_workers: Number of worker threads/processes for parallel processing
        """
        self.max_items_per_batch = max_items_per_batch
        self.parallelize = parallelize
        self.num_workers = num_workers
        
    def process_data(self, state: State) -> Dict[str, Any]:
        """
        Process retrieved data according to the current plan.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting data processing operations")
        
        if not state.current_plan:
            logger.warning("No plan to execute")
            return {"results": [], "metadata": {}}
            
        # Track execution metrics
        start_time = time.time()
        processing_results = {
            "results": [],
            "metadata": {}
        }
        
        # Execute processing steps from the plan
        for i, step in enumerate(state.current_plan.steps):
            action = step.get("action", "")
            params = step.get("params", {})
            
            # Skip non-processing steps
            if not self._is_processing_step(action):
                continue
                
            logger.info(f"Executing processing step: {action}")
            state.current_plan.mark_step_running(i)
            
            try:
                # Execute the appropriate processing action
                result = self._execute_processing_action(
                    action=action,
                    params=params,
                    state=state
                )
                
                if result:
                    # Handle different result types
                    if action == "process_results":
                        processing_results["results"] = result
                    elif action == "prepare_output":
                        processing_results["output"] = result
                    else:
                        processing_results[action] = result
                    
                    # Mark step as complete
                    state.current_plan.mark_step_complete(i, result)
                
            except Exception as e:
                logger.error(f"Error executing processing step {action}: {str(e)}", exc_info=True)
                state.current_plan.mark_step_failed(i, error=e)
                
        # Calculate execution time and add to metadata
        execution_time = time.time() - start_time
        processing_results["metadata"]["execution_time"] = execution_time
        processing_results["metadata"]["steps_executed"] = len(state.current_plan.steps)
        processing_results["metadata"]["processing_complete"] = True
        
        logger.info(f"Processing operations completed in {execution_time:.2f} seconds")
        return processing_results
    
    def _is_processing_step(self, action: str) -> bool:
        """
        Check if an action is a processing step.
        
        Args:
            action: Step action
            
        Returns:
            Boolean indicating if this is a processing step
        """
        processing_actions = {
            "process_results", 
            "prepare_output", 
            "filter_results",
            "categorize_results",
            "format_results",
            "merge_data",
            "enrich_data"
        }
        return action in processing_actions
    
    def _execute_processing_action(
        self, 
        action: str, 
        params: Dict[str, Any],
        state: State
    ) -> Any:
        """
        Execute a specific processing action.
        
        Args:
            action: Action to execute
            params: Action parameters
            state: Current workflow state
            
        Returns:
            Result of the action
        """
        if action == "process_results":
            return self._process_results(params, state)
        elif action == "prepare_output":
            return self._prepare_output(params, state)
        elif action == "filter_results":
            return self._filter_results(params, state)
        elif action == "categorize_results":
            return self._categorize_results(params, state)
        elif action == "format_results":
            return self._format_results(params, state)
        elif action == "merge_data":
            return self._merge_data(params, state)
        elif action == "enrich_data":
            return self._enrich_data(params, state)
        else:
            logger.warning(f"Unknown processing action: {action}")
            return None
    
    def _process_results(self, params: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        """
        Process and filter the results.
        
        Args:
            params: Processing parameters
            state: Current workflow state
            
        Returns:
            Processed results
        """
        sort_by = params.get("sort_by", "relevance")
        filter_duplicates = params.get("filter_duplicates", True)
        format_results = params.get("format_results", True)
        
        # Get form data from retrieval phase
        form_data = []
        if state.retrieved_data and "form_data" in state.retrieved_data:
            form_data = state.retrieved_data["form_data"]
            
        # If no form data, try to get from form references
        if not form_data and state.form_references:
            logger.info("No form data found, using form references")
            form_data = [
                {
                    "form_id": ref.form_id,
                    "title": ref.title,
                    "year": ref.year,
                    "url": ref.url,
                    "relevance_score": ref.relevance_score
                } 
                for ref in state.form_references
            ]
        
        # If still no data, return empty list
        if not form_data:
            logger.warning("No data to process")
            return []
            
        logger.info(f"Processing {len(form_data)} items")
        
        # Filter duplicates if requested
        if filter_duplicates:
            unique_results = []
            form_ids_seen = set()
            
            for result in form_data:
                form_id = result.get("form_id")
                year = result.get("year")
                key = f"{form_id}_{year}" if year else form_id
                
                if not key or key not in form_ids_seen:
                    unique_results.append(result)
                    if key:
                        form_ids_seen.add(key)
        else:
            unique_results = form_data
            
        # Sort results
        if sort_by == "relevance":
            sorted_results = sorted(
                unique_results, 
                key=lambda x: x.get("relevance_score", 0), 
                reverse=True
            )
        elif sort_by == "year":
            sorted_results = sorted(
                unique_results, 
                key=lambda x: x.get("year", 0), 
                reverse=True
            )
        elif sort_by == "form_id":
            sorted_results = sorted(
                unique_results, 
                key=lambda x: x.get("form_id", "")
            )
        else:
            sorted_results = unique_results
            
        # Format results if requested
        if format_results:
            processed_results = self._format_results(
                {"format_type": "standard"}, 
                state, 
                input_results=sorted_results
            )
        else:
            processed_results = sorted_results
            
        logger.info(f"Processed {len(form_data)} items into {len(processed_results)} results")
        
        # Update state execution stats
        state.execution_stats["processed_items"] = len(form_data)
        state.execution_stats["result_items"] = len(processed_results)
        state.execution_stats["duplicates_removed"] = len(form_data) - len(unique_results)
        
        return processed_results
    
    def _prepare_output(self, params: Dict[str, Any], state: State) -> Dict[str, Any]:
        """
        Prepare the final output from the results.
        
        Args:
            params: Output parameters
            state: Current workflow state
            
        Returns:
            Structured output
        """
        categorize = params.get("categorize", False)
        
        # Use state results or a custom input
        if "input_results" in params:
            results = params["input_results"]
        elif state.results:
            results = state.results
        else:
            logger.warning("No results to prepare output from")
            return {"total_results": 0}
            
        # Categorize results if requested
        if categorize:
            categorized = self._categorize_results(params, state, input_results=results)
            forms_by_type = categorized
        else:
            # Group by form type
            forms_by_type = defaultdict(list)
            for result in results:
                form_type = result.get("form_type", "unknown")
                forms_by_type[form_type].append(result)
                
            # Convert defaultdict to regular dict
            forms_by_type = dict(forms_by_type)
            
        # Prepare summary
        summary = {
            "query": state.query,
            "year": state.year,
            "form": state.form,
            "total_results": len(results),
            "form_types": list(forms_by_type.keys()),
            "results_by_type": forms_by_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add execution stats
        if state.execution_stats:
            summary["stats"] = state.execution_stats
            
        logger.info(f"Prepared output with {summary['total_results']} total results across {len(summary['form_types'])} form types")
        return summary
    
    def _filter_results(self, params: Dict[str, Any], state: State, input_results: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Filter results based on provided criteria.
        
        Args:
            params: Filter parameters
            state: Current workflow state
            input_results: Optional input results to filter
            
        Returns:
            Filtered results
        """
        # Get results to filter
        if input_results is not None:
            results = input_results
        elif state.results:
            results = state.results
        else:
            logger.warning("No results to filter")
            return []
            
        logger.info(f"Filtering {len(results)} results")
        
        # Apply filters
        filtered = results
        
        # Filter by form type
        if "form_type" in params:
            form_type = params["form_type"]
            filtered = [r for r in filtered if r.get("form_type") == form_type]
            
        # Filter by year
        if "year" in params:
            year = params["year"]
            filtered = [r for r in filtered if r.get("year") == year]
            
        # Filter by minimum relevance
        if "min_relevance" in params:
            min_relevance = params["min_relevance"]
            filtered = [r for r in filtered if r.get("relevance_score", 0) >= min_relevance]
            
        # Filter by form ID pattern
        if "form_id_pattern" in params:
            import re
            pattern = re.compile(params["form_id_pattern"], re.IGNORECASE)
            filtered = [r for r in filtered if pattern.search(r.get("form_id", ""))]
            
        logger.info(f"Filtered to {len(filtered)} results")
        return filtered
    
    def _categorize_results(self, params: Dict[str, Any], state: State, input_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize results into logical groups.
        
        Args:
            params: Categorization parameters
            state: Current workflow state
            input_results: Optional input results to categorize
            
        Returns:
            Categorized results
        """
        # Get results to categorize
        if input_results is not None:
            results = input_results
        elif state.results:
            results = state.results
        else:
            logger.warning("No results to categorize")
            return {}
            
        logger.info(f"Categorizing {len(results)} results")
        
        # Define categorization function
        categorize_by = params.get("categorize_by", "form_type")
        
        if categorize_by == "form_type":
            # Map form IDs to categories based on pattern matching
            categories = defaultdict(list)
            
            for result in results:
                form_id = result.get("form_id", "").upper()
                
                if form_id.startswith("1040"):
                    category = "individual_income_tax"
                elif form_id.startswith("1099"):
                    category = "information_returns"
                elif form_id.startswith("W-"):
                    category = "wage_reporting"
                elif form_id.startswith("941") or form_id.startswith("940"):
                    category = "employment_tax"
                elif form_id.startswith("1120"):
                    category = "corporate_tax"
                elif form_id.startswith("1065"):
                    category = "partnership_tax"
                elif form_id.startswith("990"):
                    category = "nonprofit_returns"
                elif form_id.startswith("706") or form_id.startswith("709"):
                    category = "estate_gift_tax"
                elif "SCH" in form_id or "SCHEDULE" in form_id:
                    category = "schedules"
                else:
                    category = "other"
                    
                categories[category].append(result)
                
        elif categorize_by == "year":
            # Group by year
            categories = defaultdict(list)
            
            for result in results:
                year = result.get("year", "unknown")
                categories[str(year)].append(result)
                
        elif categorize_by == "relevance":
            # Group by relevance range
            categories = {
                "high": [],
                "medium": [],
                "low": []
            }
            
            for result in results:
                score = result.get("relevance_score", 0)
                
                if score >= 0.7:
                    categories["high"].append(result)
                elif score >= 0.4:
                    categories["medium"].append(result)
                else:
                    categories["low"].append(result)
                    
        else:
            # Default grouping by form_type field
            categories = defaultdict(list)
            
            for result in results:
                category = result.get(categorize_by, "unknown")
                categories[category].append(result)
                
        # Convert defaultdict to regular dict
        categorized = dict(categories)
        
        logger.info(f"Categorized into {len(categorized)} groups")
        return categorized
    
    def _format_results(self, params: Dict[str, Any], state: State, input_results: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Format results according to specified format.
        
        Args:
            params: Format parameters
            state: Current workflow state
            input_results: Optional input results to format
            
        Returns:
            Formatted results
        """
        # Get results to format
        if input_results is not None:
            results = input_results
        elif state.results:
            results = state.results
        elif state.retrieved_data and "form_data" in state.retrieved_data:
            results = state.retrieved_data["form_data"]
        else:
            logger.warning("No results to format")
            return []
            
        logger.info(f"Formatting {len(results)} results")
        
        format_type = params.get("format_type", "standard")
        
        if format_type == "standard":
            # Standard format with consistent fields
            formatted = []
            
            for result in results:
                # Create a standardized result object
                standard_result = {
                    "form_id": result.get("form_id", ""),
                    "title": result.get("title", ""),
                    "form_type": self._determine_form_type(result),
                    "year": result.get("year"),
                    "url": result.get("url", ""),
                    "instructions_url": result.get("instructions_url", ""),
                    "relevance_score": result.get("relevance_score", 0.5)
                }
                
                # Add filing deadline if available
                if "filing_deadline" in result:
                    standard_result["filing_deadline"] = result["filing_deadline"]
                    
                # Add related forms if available
                if "related_forms" in result:
                    standard_result["related_forms"] = result["related_forms"]
                    
                formatted.append(standard_result)
                
        elif format_type == "minimal":
            # Minimal format with only essential fields
            formatted = [
                {
                    "form_id": result.get("form_id", ""),
                    "title": result.get("title", ""),
                    "year": result.get("year"),
                    "url": result.get("url", "")
                }
                for result in results
            ]
            
        elif format_type == "detailed":
            # Detailed format with all available fields
            formatted = results
            
        else:
            # Unknown format, return as is
            logger.warning(f"Unknown format type: {format_type}")
            formatted = results
            
        logger.info(f"Formatted {len(results)} results")
        return formatted
    
    def _merge_data(self, params: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        """
        Merge data from multiple sources.
        
        Args:
            params: Merge parameters
            state: Current workflow state
            
        Returns:
            Merged results
        """
        # Get primary results
        primary_results = state.results
        
        # Get secondary data sources
        secondary_sources = []
        
        if state.retrieved_data and "form_data" in state.retrieved_data:
            secondary_sources.append(("form_data", state.retrieved_data["form_data"]))
            
        if state.form_references:
            secondary_sources.append(("form_refs", [ref.dict() for ref in state.form_references]))
            
        if not primary_results and not secondary_sources:
            logger.warning("No data to merge")
            return []
            
        logger.info(f"Merging data from {len(secondary_sources) + 1} sources")
        
        # Create a map of form_id -> result for primary results
        result_map = {}
        for result in primary_results:
            form_id = result.get("form_id")
            year = result.get("year")
            key = f"{form_id}_{year}" if year else form_id
            
            if key:
                result_map[key] = result.copy()
                
        # Merge in data from secondary sources
        for source_name, source_data in secondary_sources:
            for item in source_data:
                form_id = item.get("form_id")
                year = item.get("year")
                key = f"{form_id}_{year}" if year else form_id
                
                if not key:
                    continue
                    
                if key in result_map:
                    # Merge fields that don't exist in the primary result
                    for field, value in item.items():
                        if field not in result_map[key] or result_map[key][field] is None:
                            result_map[key][field] = value
                else:
                    # Add to results if not already present
                    result_map[key] = item.copy()
                    
        # Convert back to list
        merged_results = list(result_map.values())
        
        logger.info(f"Merged into {len(merged_results)} results")
        return merged_results
    
    def _enrich_data(self, params: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        """
        Enrich data with additional information.
        
        Args:
            params: Enrichment parameters
            state: Current workflow state
            
        Returns:
            Enriched results
        """
        # Get results to enrich
        if state.results:
            results = state.results.copy()
        else:
            logger.warning("No results to enrich")
            return []
            
        logger.info(f"Enriching {len(results)} results")
        
        # Determine which enrichments to apply
        add_links = params.get("add_links", True)
        add_form_types = params.get("add_form_types", True)
        add_descriptions = params.get("add_descriptions", False)
        
        # Use parallel processing if enabled
        if self.parallelize and len(results) > 10:
            # Define enrichment function
            def enrich_item(item):
                enriched_item = item.copy()
                
                if add_links and "form_id" in item:
                    enriched_item = self._add_links(enriched_item)
                    
                if add_form_types and "form_type" not in item:
                    enriched_item["form_type"] = self._determine_form_type(item)
                    
                if add_descriptions and "description" not in item:
                    enriched_item["description"] = self._generate_description(item)
                    
                return enriched_item
                
            # Process in parallel
            enriched_results = parallel_process(
                items=results,
                process_func=enrich_item,
                max_workers=self.num_workers
            )
            
        else:
            # Process sequentially
            enriched_results = []
            
            for item in results:
                enriched_item = item.copy()
                
                if add_links and "form_id" in item:
                    enriched_item = self._add_links(enriched_item)
                    
                if add_form_types and "form_type" not in item:
                    enriched_item["form_type"] = self._determine_form_type(item)
                    
                if add_descriptions and "description" not in item:
                    enriched_item["description"] = self._generate_description(item)
                    
                enriched_results.append(enriched_item)
                
        logger.info(f"Enriched {len(results)} results")
        return enriched_results
    
    def _determine_form_type(self, result: Dict[str, Any]) -> str:
        """
        Determine the form type based on form ID and other metadata.
        
        Args:
            result: Form data
            
        Returns:
            Form type string
        """
        # If form_type is already set, use it
        if "form_type" in result and result["form_type"]:
            return result["form_type"]
            
        form_id = result.get("form_id", "").upper()
        
        # Map form IDs to categories based on pattern matching
        if form_id.startswith("1040"):
            return "individual_income_tax"
        elif form_id.startswith("1099"):
            return "information_returns"
        elif form_id.startswith("W-"):
            return "wage_reporting"
        elif form_id.startswith("941") or form_id.startswith("940"):
            return "employment_tax"
        elif form_id.startswith("1120"):
            return "corporate_tax"
        elif form_id.startswith("1065"):
            return "partnership_tax"
        elif form_id.startswith("990"):
            return "nonprofit_returns"
        elif form_id.startswith("706") or form_id.startswith("709"):
            return "estate_gift_tax"
        elif "SCH" in form_id or "SCHEDULE" in form_id:
            return "schedules"
        else:
            return "other"
    
    def _add_links(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add useful links to a result.
        
        Args:
            result: Form data
            
        Returns:
            Form data with added links
        """
        form_id = result.get("form_id", "").replace(" ", "")
        year = result.get("year", datetime.now().year)
        
        # Add standard IRS website links if not present
        if "url" not in result or not result["url"]:
            result["url"] = f"https://www.irs.gov/forms/{form_id.lower()}"
            
        if "instructions_url" not in result or not result["instructions_url"]:
            result["instructions_url"] = f"https://www.irs.gov/instructions/{form_id.lower()}"
            
        # Add PDF links
        if "pdf_url" not in result:
            result["pdf_url"] = f"https://www.irs.gov/pub/irs-pdf/{form_id.lower()}.pdf"
            
        if "instructions_pdf_url" not in result:
            result["instructions_pdf_url"] = f"https://www.irs.gov/pub/irs-pdf/i{form_id.lower()}.pdf"
            
        return result
    
    def _generate_description(self, result: Dict[str, Any]) -> str:
        """
        Generate a descriptive summary for a form.
        
        Args:
            result: Form data
            
        Returns:
            Description string
        """
        form_id = result.get("form_id", "")
        title = result.get("title", "")
        form_type = result.get("form_type", self._determine_form_type(result))
        year = result.get("year", "")
        
        # Create a basic description
        description = f"IRS {form_id}"
        
        if title:
            description += f": {title}"
            
        if year:
            description += f" for tax year {year}"
            
        # Add type-specific descriptions
        if form_type == "individual_income_tax":
            description += ". Used for filing individual income tax returns."
        elif form_type == "information_returns":
            description += ". Used for reporting various types of income other than wages."
        elif form_type == "wage_reporting":
            description += ". Used for reporting wage and tax information."
        elif form_type == "employment_tax":
            description += ". Used for reporting employment taxes."
        elif form_type == "corporate_tax":
            description += ". Used for filing corporate income tax returns."
        elif form_type == "partnership_tax":
            description += ". Used for reporting partnership income."
        elif form_type == "nonprofit_returns":
            description += ". Used by tax-exempt organizations for annual reporting."
        elif form_type == "estate_gift_tax":
            description += ". Used for estate or gift tax reporting."
            
        return description
