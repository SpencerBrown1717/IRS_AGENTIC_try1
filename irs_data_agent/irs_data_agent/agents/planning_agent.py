"""
Planning agent for IRS data retrieval.
"""
from typing import List, Dict, Any
from pydantic import BaseModel

from irs_data_agent.state import State

class Step(BaseModel):
    """A single step in a retrieval plan."""
    name: str
    description: str
    parameters: Dict[str, Any] = {}

class Plan(BaseModel):
    """A plan for retrieving IRS data."""
    steps: List[Step] = []
    description: str

class PlanningAgent:
    """
    Agent responsible for creating plans to retrieve IRS data based on user queries.
    """
    
    def create_plan(self, state: State) -> Plan:
        """
        Create a plan for retrieving IRS data based on the current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Plan object with steps to execute
        """
        # This is a simplified implementation
        # In a real system, this might use LLMs or other AI techniques
        
        plan = Plan(description=f"Plan to retrieve IRS data for query: {state.query}")
        
        # Add steps based on the query and parameters
        if state.form:
            # If a specific form is requested
            plan.steps.append(
                Step(
                    name="retrieve_form",
                    description=f"Retrieve IRS form {state.form}",
                    parameters={"form_number": state.form, "year": state.year}
                )
            )
        else:
            # General search query
            plan.steps.append(
                Step(
                    name="search_irs_database",
                    description=f"Search IRS database for: {state.query}",
                    parameters={"query": state.query, "year": state.year}
                )
            )
            
            plan.steps.append(
                Step(
                    name="filter_results",
                    description="Filter and rank results by relevance",
                    parameters={"min_relevance_score": 0.7}
                )
            )
        
        # Always add a formatting step
        plan.steps.append(
            Step(
                name="format_results",
                description="Format results for presentation",
                parameters={}
            )
        )
        
        return plan
