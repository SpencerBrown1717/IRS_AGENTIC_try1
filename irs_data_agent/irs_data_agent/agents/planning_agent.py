"""
Enhanced planning agent for IRS data retrieval.
"""
from typing import Dict, List, Any, Optional, Set
from irs_data_agent.core.state import State, Plan, AnalysisResult
from irs_data_agent.utils.logging import get_logger
import re

logger = get_logger(__name__)

class PlanningAgent:
    """
    Enhanced agent responsible for creating plans to retrieve and process IRS data.
    """
    
    def __init__(self, max_steps: int = 10):
        """
        Initialize the planning agent.
        
        Args:
            max_steps: Maximum number of steps in a plan
        """
        self.max_steps = max_steps
        
    def create_plan(self, state: State) -> Plan:
        """
        Create a plan based on the current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            A plan object with steps to execute
        """
        logger.info(f"Creating plan for query: {state.query}")
        plan = Plan()
        
        # Step 1: Analyze the query to understand intent
        plan.add_step(
            action="analyze_query",
            params={"query": state.query},
            description="Analyze query to determine search parameters"
        )
        
        # Step 2: Search for relevant forms
        search_params = {"keywords": state.query}
        if state.form:
            search_params["form_type"] = state.form
        if state.year:
            search_params["year"] = state.year
            
        plan.add_step(
            action="search_forms",
            params=search_params,
            description="Search for relevant IRS forms"
        )
        
        # Step 3: Filter and rank the search results
        plan.add_step(
            action="rank_search_results",
            params={
                "min_relevance": 0.6,
                "max_results": 10
            },
            description="Filter and rank search results by relevance"
        )
        
        # Step 4: Fetch data for each relevant form
        plan.add_step(
            action="fetch_form_data",
            params={"max_forms": 10, "include_metadata": True},
            description="Fetch detailed data for each relevant form"
        )
        
        # Step 5: Check if additional data is needed
        plan.add_step(
            action="evaluate_data_completeness",
            params={},
            description="Check if additional data is needed"
        )
        
        # Step 6: Process the results
        plan.add_step(
            action="process_results",
            params={
                "sort_by": "relevance",
                "filter_duplicates": True,
                "format_results": True
            },
            description="Process and format the collected data"
        )
        
        # Step 7: Prepare final output
        plan.add_step(
            action="prepare_output",
            params={"categorize": True},
            description="Prepare final structured output"
        )
        
        logger.info(f"Created plan with {len(plan.steps)} steps")
        return plan
    
    def analyze_query(self, query: str) -> AnalysisResult:
        """
        Analyze a query to determine search parameters.
        
        Args:
            query: Search query
            
        Returns:
            AnalysisResult with analysis information
        """
        logger.info(f"Analyzing query: {query}")
        
        # Initialize result
        result = AnalysisResult(
            possible_form_types=[],
            possible_years=[],
            keywords=[],
            entities={}
        )
        
        # Normalize query
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Extract possible years (2000-2030)
        year_pattern = re.compile(r'\b(20[0-2][0-9])\b')
        years = year_pattern.findall(query)
        result.possible_years = [int(year) for year in years]
        
        # Extract possible form types
        form_patterns = [
            (r'\bform\s+(\w+[-\w]*)', 'form'),
            (r'\bschedule\s+([a-z0-9]+)', 'schedule'),
            (r'\b(1040|1099|w-[24]|990|941|1120)\b', 'form')
        ]
        
        for pattern, form_type in form_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                if len(match.groups()) > 0:
                    form_id = match.group(1).upper()
                    result.possible_form_types.append(form_id)
                    result.entities[form_id] = form_type
                else:
                    form_id = match.group(0).upper()
                    result.possible_form_types.append(form_id)
                    result.entities[form_id] = form_type
        
        # Extract keywords (exclude stop words and form/year terms)
        stop_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
            'about', 'form', 'irs', 'tax', 'year', 'schedule', 'and', 'or'
        }
        
        form_related_words = {'form', 'schedule', 'publication', 'instructions'}
        year_words = {str(year) for year in range(2000, 2030)}
        
        # Create set of words to exclude
        exclude_words = stop_words.union(form_related_words).union(year_words)
        exclude_words.update(result.possible_form_types)
        
        # Extract keywords (words not in exclude_words)
        result.keywords = [word for word in words 
                          if word not in exclude_words and len(word) > 2]
        
        # Set confidence based on extracted information
        if result.possible_form_types and result.possible_years:
            result.confidence = 0.9
        elif result.possible_form_types:
            result.confidence = 0.7
        elif result.possible_years:
            result.confidence = 0.6
        else:
            result.confidence = 0.4
            
        logger.info(f"Query analysis results: {result.dict()}")
        return result
    
    def refine_plan(self, state: State) -> Plan:
        """
        Refine an existing plan based on the current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            A refined plan
        """
        logger.info("Refining plan based on current state")
        
        # Check if we have results
        has_results = bool(state.results)
        has_form_references = bool(state.form_references)
        has_analysis = state.query_analysis is not None
        
        # Create a new plan
        plan = Plan()
        
        if not has_analysis:
            # If no analysis yet, start with that
            plan.add_step(
                action="analyze_query",
                params={"query": state.query},
                description="Analyze query to determine search parameters"
            )
        
        if not has_form_references:
            # If no form references, search for forms
            search_params = {"keywords": state.query}
            if state.form:
                search_params["form_type"] = state.form
            if state.year:
                search_params["year"] = state.year
                
            plan.add_step(
                action="search_forms",
                params=search_params,
                description="Search for relevant IRS forms"
            )
            
            # Then rank them
            plan.add_step(
                action="rank_search_results",
                params={"min_relevance": 0.5},
                description="Filter and rank search results by relevance"
            )
        
        if has_form_references and not has_results:
            # If we have form references but no results, fetch data
            plan.add_step(
                action="fetch_form_data",
                params={"max_forms": 5},
                description="Fetch detailed data for each relevant form"
            )
        
        if not has_results:
            # If no results yet, try a broader search
            plan.add_step(
                action="broaden_search",
                params={"query": state.query},
                description="Broaden search to find more results"
            )
        else:
            # Otherwise, process what we have
            plan.add_step(
                action="process_results",
                params={"sort_by": "relevance"},
                description="Process and format the collected data"
            )
            
        # Always add output preparation
        plan.add_step(
            action="prepare_output",
            params={},
            description="Prepare final structured output"
        )
        
        return plan
    
    def create_recovery_plan(self, state: State, error: Dict[str, Any]) -> Optional[Plan]:
        """
        Create a recovery plan after an error.
        
        Args:
            state: Current workflow state
            error: Error information
            
        Returns:
            A recovery plan or None if recovery not possible
        """
        logger.info(f"Creating recovery plan for error: {error.get('type', 'unknown')}")
        
        error_type = error.get('type', '')
        error_message = error.get('message', '')
        
        # Create a recovery plan based on error type
        plan = Plan()
        
        if 'timeout' in error_type.lower() or 'timeout' in error_message.lower():
            # Handle timeout errors
            plan.add_step(
                action="adjust_timeouts",
                params={"increase_factor": 1.5},
                description="Increase timeouts for API requests"
            )
            
            plan.add_step(
                action="retry_operation",
                params={"max_retries": 3, "backoff_factor": 2},
                description="Retry the failed operation with backoff"
            )
            
        elif 'rate' in error_type.lower() or 'limit' in error_message.lower():
            # Handle rate limit errors
            plan.add_step(
                action="implement_rate_limiting",
                params={"requests_per_minute": 10},
                description="Apply stricter rate limiting"
            )
            
            plan.add_step(
                action="wait_and_retry",
                params={"wait_seconds": 60},
                description="Wait and retry the operation"
            )
            
        elif 'not found' in error_message.lower() or '404' in error_message:
            # Handle not found errors
            plan.add_step(
                action="broaden_search",
                params={"query": state.query, "fuzzy_match": True},
                description="Try a broader search with fuzzy matching"
            )
            
        else:
            # Generic recovery plan
            plan.add_step(
                action="simplify_request",
                params={},
                description="Simplify the request parameters"
            )
            
            plan.add_step(
                action="retry_operation",
                params={"max_retries": 2},
                description="Retry the operation with simplified parameters"
            )
        
        # Add final step to check results
        plan.add_step(
            action="evaluate_results",
            params={},
            description="Evaluate the results after recovery"
        )
        
        return plan if len(plan.steps) > 0 else None