"""
A simple workflow engine for orchestrating IRS data retrieval and processing.
"""
from typing import Dict, Any, List
from rich.console import Console

from irs_data_agent.state import State
from irs_data_agent.agents.planning_agent import PlanningAgent
from irs_data_agent.agents.execution_agent import ExecutionAgent
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

class Workflow:
    """
    Coordinates the IRS data retrieval workflow using planning and execution agents.
    """
    
    def __init__(
        self,
        state: State,
        planning_agent: PlanningAgent,
        execution_agent: ExecutionAgent
    ):
        """
        Initialize the workflow with state and agents.
        
        Args:
            state: The current state of the workflow
            planning_agent: Agent responsible for creating retrieval plans
            execution_agent: Agent responsible for executing retrieval plans
        """
        self.state = state
        self.planning_agent = planning_agent
        self.execution_agent = execution_agent
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the workflow to retrieve IRS data.
        
        Returns:
            List of result dictionaries
        """
        try:
            # Step 1: Generate a plan using the planning agent
            console.print("[bold]Step 1:[/] Generating retrieval plan...")
            plan = self.planning_agent.create_plan(self.state)
            console.print(f"[green]✓[/] Plan created with {len(plan.steps)} steps")
            
            # Step 2: Execute the plan using the execution agent
            console.print("[bold]Step 2:[/] Executing retrieval plan...")
            results = self.execution_agent.execute_plan(plan, self.state)
            console.print(f"[green]✓[/] Plan executed, retrieved {len(results)} results")
            
            # Step 3: Update state with results
            self.state.results = results
            self.state.mark_complete()
            
            return results
            
        except Exception as e:
            console.print(f"[bold red]Error in workflow:[/] {str(e)}")
            self.state.set_error(str(e))
            return []
