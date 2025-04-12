"""
Execution agent for IRS data retrieval.
"""
from typing import List, Dict, Any
import time
from rich.console import Console

from irs_data_agent.state import State
from irs_data_agent.agents.planning_agent import Plan, Step
from irs_data_agent.api.irs_client import IRSClient

console = Console()

class ExecutionAgent:
    """
    Agent responsible for executing plans to retrieve IRS data.
    """
    
    def __init__(self, client: IRSClient):
        """
        Initialize the execution agent.
        
        Args:
            client: IRS API client for making requests
        """
        self.client = client
    
    def execute_plan(self, plan: Plan, state: State) -> List[Dict[str, Any]]:
        """
        Execute a plan to retrieve IRS data.
        
        Args:
            plan: Plan to execute
            state: Current workflow state
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, step in enumerate(plan.steps):
            console.print(f"[bold]Executing step {i+1}/{len(plan.steps)}:[/] {step.name}")
            console.print(f"  {step.description}")
            
            # Execute the step based on its name
            step_results = self._execute_step(step, state)
            
            # Add step results to overall results if not empty
            if step_results:
                results.extend(step_results)
                console.print(f"  [green]✓[/] Step completed with {len(step_results)} results")
            else:
                console.print(f"  [yellow]![/] Step completed with no results")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def _execute_step(self, step: Step, state: State) -> List[Dict[str, Any]]:
        """
        Execute a single step in the plan.
        
        Args:
            step: Step to execute
            state: Current workflow state
            
        Returns:
            Results from executing the step
        """
        # Execute different actions based on step name
        if step.name == "search_irs_database":
            return self.client.search(
                query=step.parameters.get("query", ""),
                year=step.parameters.get("year")
            )
        
        elif step.name == "retrieve_form":
            form = self.client.get_form(
                form_number=step.parameters.get("form_number", ""),
                year=step.parameters.get("year")
            )
            return [form] if form else []
        
        elif step.name == "filter_results":
            # Filter the existing results in the state
            min_score = step.parameters.get("min_relevance_score", 0.5)
            return [r for r in state.results if r.get("relevance_score", 0) >= min_score]
        
        elif step.name == "format_results":
            # This step doesn't add new results, just processes existing ones
            return []
        
        else:
            console.print(f"[red]Unknown step: {step.name}[/]")
            return []
