"""
Retrieval agent for IRS data fetching operations.
"""
from typing import Dict, List, Any, Optional
from irs_data_agent.core.state import State, FormReference
from irs_data_agent.api.irs_client import IRSClient
from irs_data_agent.api.rate_limiter import RateLimiter
from irs_data_agent.utils.logging import get_logger
from irs_data_agent.utils.parallel import parallel_process

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = get_logger(__name__)

class RetrievalAgent:
    """
    Agent responsible for retrieving IRS data according to search parameters.
    """
    
    def __init__(
        self, 
        client: IRSClient,
        batch_size: int = 5,
        max_concurrent_requests: int = 3,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize the retrieval agent.
        
        Args:
            client: IRS API client
            batch_size: Number of items to process in a batch
            max_concurrent_requests: Maximum number of concurrent requests
            rate_limiter: Optional rate limiter
        """
        self.client = client
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limiter = rate_limiter or RateLimiter()
        
    def execute_retrieval(self, state: State) -> Dict[str, Any]:
        """
        Execute retrieval operations based on the current plan.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary with retrieval results
        """
        logger.info("Starting retrieval operations")
        
        if not state.current_plan:
            logger.warning("No plan to execute")
            return {}
            
        # Track execution metrics
        start_time = time.time()
        retrieval_results = {
            "form_data": [],
            "metadata": {}
        }
        
        # Execute retrieval steps from the plan
        for i, step in enumerate(state.current_plan.steps):
            action = step.get("action", "")
            params = step.get("params", {})
            
            # Skip non-retrieval steps
            if not self._is_retrieval_step(action):
                continue
                
            logger.info(f"Executing retrieval step: {action}")
            state.current_plan.mark_step_running(i)
            
            try:
                # Execute the appropriate retrieval action
                result = self._execute_retrieval_action(
                    action=action,
                    params=params,
                    state=state
                )
                
                if result:
                    # Add result to overall results
                    if action == "analyze_query":
                        state.update_with_analysis(result)
                        retrieval_results["query_analysis"] = result
                    elif action == "search_forms":
                        self._process_search_results(state, result)
                        retrieval_results["search_results"] = result
                    elif action == "fetch_form_data":
                        retrieval_results["form_data"].extend(result)
                    
                    # Mark step as complete
                    state.current_plan.mark_step_complete(i, result)
                
            except Exception as e:
                logger.error(f"Error executing retrieval step {action}: {str(e)}", exc_info=True)
                state.current_plan.mark_step_failed(i, error=e)
                
        # Calculate execution time and add to metadata
        execution_time = time.time() - start_time
        retrieval_results["metadata"]["execution_time"] = execution_time
        retrieval_results["metadata"]["steps_executed"] = len(state.current_plan.steps)
        retrieval_results["metadata"]["retrieval_complete"] = True
        
        logger.info(f"Retrieval operations completed in {execution_time:.2f} seconds")
        return retrieval_results
    
    def _is_retrieval_step(self, action: str) -> bool:
        """
        Check if an action is a retrieval step.
        
        Args:
            action: Step action
            
        Returns:
            Boolean indicating if this is a retrieval step
        """
        retrieval_actions = {
            "analyze_query", 
            "search_forms", 
            "fetch_form_data",
            "broaden_search",
            "evaluate_data_completeness",
            "rank_search_results"
        }
        return action in retrieval_actions
    
    def _execute_retrieval_action(
        self, 
        action: str, 
        params: Dict[str, Any],
        state: State
    ) -> Any:
        """
        Execute a specific retrieval action.
        
        Args:
            action: Action to execute
            params: Action parameters
            state: Current workflow state
            
        Returns:
            Result of the action
        """
        # Apply rate limiting
        self.rate_limiter.limit()
        
        if action == "analyze_query":
            return self._analyze_query(params, state)
        elif action == "search_forms":
            return self._search_forms(params, state)
        elif action == "fetch_form_data":
            return self._fetch_form_data(params, state)
        elif action == "broaden_search":
            return self._broaden_search(params, state)
        elif action == "evaluate_data_completeness":
            return self._evaluate_data_completeness(params, state)
        elif action == "rank_search_results":
            return self._rank_search_results(params, state)
        else:
            logger.warning(f"Unknown retrieval action: {action}")
            return None
    
    def _analyze_query(self, params: Dict[str, Any], state: State) -> Dict[str, Any]:
        """
        Analyze the query to determine search parameters.
        
        Args:
            params: Query parameters
            state: Current workflow state
            
        Returns:
            Analysis results
        """
        query = params.get("query", state.query)
        logger.info(f"Analyzing query: {query}")
        
        # If the planning agent has an analyze_query method, use that
        if hasattr(state, 'planning_agent') and hasattr(state.planning_agent, 'analyze_query'):
            return state.planning_agent.analyze_query(query).dict()
        
        # Otherwise use a simpler approach
        words = query.lower().split()
        
        analysis = {
            "query": query,
            "possible_form_types": [],
            "possible_years": [],
            "keywords": [],
            "entities": {}
        }
        
        # Simple pattern matching for common form types
        form_prefixes = ["form", "schedule", "publication"]
        years = [str(year) for year in range(2000, 2030)]
        
        for word in words:
            # Check for form types
            for prefix in form_prefixes:
                if word.startswith(prefix):
                    analysis["possible_form_types"].append(word)
                    analysis["entities"][word] = "form"
                    
            # Check for years
            if word in years:
                analysis["possible_years"].append(int(word))
                
            # Add remaining words as keywords
            if word not in analysis["possible_form_types"] and word not in years:
                analysis["keywords"].append(word)
        
        logger.info(f"Query analysis: {analysis}")
        return analysis
    
    def _search_forms(self, params: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        """
        Search for relevant IRS forms.
        
        Args:
            params: Search parameters
            state: Current workflow state
            
        Returns:
            List of matching forms
        """
        keywords = params.get("keywords", state.query)
        form_type = params.get("form_type", state.form)
        year = params.get("year", state.year)
        limit = params.get("limit", 20)
        
        logger.info(f"Searching forms with keywords: {keywords}")
        
        # Use the client to search forms
        forms = self.client.search_forms(
            keywords=keywords,
            form_type=form_type,
            year=year,
            limit=limit
        )
        
        logger.info(f"Found {len(forms)} relevant forms")
        return forms
    
    def _fetch_form_data(self, params: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        """
        Fetch data for forms identified in previous steps.
        
        Args:
            params: Fetch parameters
            state: Current workflow state
            
        Returns:
            List of form data
        """
        max_forms = params.get("max_forms", 10)
        include_metadata = params.get("include_metadata", True)
        
        # Get form references from state
        forms_to_fetch = state.form_references[:max_forms]
        
        if not forms_to_fetch and state.retrieved_data and "search_results" in state.retrieved_data:
            # Create form references from search results
            search_results = state.retrieved_data["search_results"]
            for result in search_results[:max_forms]:
                form_ref = FormReference(
                    form_id=result.get("form_id", ""),
                    title=result.get("title", ""),
                    year=result.get("year"),
                    url=result.get("url")
                )
                forms_to_fetch.append(form_ref)
                
        logger.info(f"Fetching data for {len(forms_to_fetch)} forms")
        
        # Use parallel processing to fetch data faster
        form_data = []
        
        # Define fetch function for parallel processing
        def fetch_form(form):
            try:
                # Apply rate limiting
                self.rate_limiter.limit()
                
                if isinstance(form, FormReference):
                    form_id = form.form_id
                    year = form.year
                elif isinstance(form, dict):
                    form_id = form.get("form_id", "")
                    year = form.get("year")
                else:
                    form_id = str(form)
                    year = None
                    
                data = self.client.get_form_data(form_id=form_id, year=year)
                
                # Add metadata if requested
                if include_metadata:
                    data["_metadata"] = {
                        "fetched_at": time.time()
                    }
                    
                return data
            except Exception as e:
                logger.error(f"Error fetching data for form {form}: {str(e)}")
                return None
        
        # Use parallel processing
        if self.max_concurrent_requests > 1:
            results = parallel_process(
                items=forms_to_fetch,
                process_func=fetch_form,
                max_workers=self.max_concurrent_requests
            )
            
            # Filter out None results
            form_data = [r for r in results if r is not None]
        else:
            # Sequential processing
            for form in forms_to_fetch:
                result = fetch_form(form)
                if result is not None:
                    form_data.append(result)
                
        logger.info(f"Fetched data for {len(form_data)} forms")
        return form_data
    
    def _broaden_search(self, params: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        """
        Perform a broader search when initial results are insufficient.
        
        Args:
            params: Search parameters
            state: Current workflow state
            
        Returns:
            List of search results
        """
        query = params.get("query", state.query)
        fuzzy_match = params.get("fuzzy_match", False)
        
        # Extract keywords
        if state.query_analysis and state.query_analysis.keywords:
            keywords = state.query_analysis.keywords
        else:
            keywords = [word for word in query.lower().split() if len(word) > 3]
        
        # Try a broader search with just keywords
        if keywords:
            logger.info(f"Broadening search with keywords: {keywords}")
            
            # Use a subset of keywords for broader match
            search_keywords = " ".join(keywords[:3])
            
            return self.client.search_forms(
                keywords=search_keywords,
                fuzzy_match=fuzzy_match,
                limit=20
            )
        else:
            return []
            
    def _evaluate_data_completeness(self, params: Dict[str, Any], state: State) -> Dict[str, Any]:
        """
        Evaluate if the retrieved data is complete or needs additional queries.
        
        Args:
            params: Evaluation parameters
            state: Current workflow state
            
        Returns:
            Evaluation results
        """
        # Check if we have enough form data
        form_data_count = len(state.retrieved_data.get("form_data", []))
        search_results_count = len(state.retrieved_data.get("search_results", []))
        
        # Calculate completeness metrics
        completeness = {
            "has_form_data": form_data_count > 0,
            "has_search_results": search_results_count > 0,
            "form_data_count": form_data_count,
            "search_results_count": search_results_count,
            "is_complete": form_data_count >= params.get("min_forms", 1),
            "coverage_ratio": form_data_count / max(1, search_results_count),
            "needs_more_data": False
        }
        
        # Determine if we need more data
        if completeness["coverage_ratio"] < 0.5 and search_results_count > form_data_count:
            completeness["needs_more_data"] = True
            completeness["recommendation"] = "fetch_more_forms"
            
        logger.info(f"Data completeness evaluation: {completeness}")
        return completeness
    
    def _rank_search_results(self, params: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        """
        Rank and filter search results by relevance.
        
        Args:
            params: Ranking parameters
            state: Current workflow state
            
        Returns:
            Ranked and filtered search results
        """
        min_relevance = params.get("min_relevance", 0.0)
        max_results = params.get("max_results", 20)
        
        if not state.retrieved_data or "search_results" not in state.retrieved_data:
            logger.warning("No search results to rank")
            return []
            
        search_results = state.retrieved_data["search_results"]
        
        # Calculate relevance scores if not already present
        for result in search_results:
            if "relevance_score" not in result:
                result["relevance_score"] = self._calculate_relevance_score(result, state)
                
        # Filter by minimum relevance
        filtered_results = [
            result for result in search_results 
            if result.get("relevance_score", 0) >= min_relevance
        ]
        
        # Sort by relevance score
        ranked_results = sorted(
            filtered_results, 
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )
        
        # Limit the number of results
        top_results = ranked_results[:max_results]
        
        logger.info(f"Ranked {len(search_results)} results, kept {len(top_results)} with relevance >= {min_relevance}")
        
        # Create form references for top results
        self._process_search_results(state, top_results)
        
        return top_results
    
    def _calculate_relevance_score(self, result: Dict[str, Any], state: State) -> float:
        """
        Calculate a relevance score for a search result.
        
        Args:
            result: Search result
            state: Current workflow state
            
        Returns:
            Relevance score between 0 and 1
        """
        score = 0.5  # Default middle score
        
        # Exact form match is highly relevant
        if state.form and result.get("form_id", "").lower() == state.form.lower():
            score += 0.3
            
        # Year match is relevant
        if state.year and result.get("year") == state.year:
            score += 0.2
            
        # Check if form type matches any in the query analysis
        if state.query_analysis and state.query_analysis.possible_form_types:
            if any(form_type.lower() in result.get("form_id", "").lower() 
                   for form_type in state.query_analysis.possible_form_types):
                score += 0.2
                
        # Check for keyword matches in title
        if state.query_analysis and state.query_analysis.keywords:
            title = result.get("title", "").lower()
            keyword_matches = sum(1 for keyword in state.query_analysis.keywords if keyword in title)
            score += 0.1 * min(keyword_matches, 3) / 3
            
        # Cap at 1.0
        return min(1.0, score)
    
    def _process_search_results(self, state: State, results: List[Dict[str, Any]]):
        """
        Process search results and update state with form references.
        
        Args:
            state: Current workflow state
            results: Search results to process
        """
        form_references = []
        
        for result in results:
            form_ref = FormReference(
                form_id=result.get("form_id", ""),
                title=result.get("title", ""),
                year=result.get("year"),
                url=result.get("url"),
                relevance_score=result.get("relevance_score", 0.5)
            )
            form_references.append(form_ref)
            
        # Update state with form references
        state.add_form_references(form_references)
