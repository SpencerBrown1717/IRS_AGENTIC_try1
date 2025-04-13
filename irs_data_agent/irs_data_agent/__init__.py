"""
IRS Data Agent - A toolkit for interacting with IRS data using intelligent agents.
"""

__version__ = "0.1.0"

from irs_data_agent.workflow import Workflow
from irs_data_agent.state import State
from irs_data_agent.agents.planning_agent import PlanningAgent
from irs_data_agent.agents.execution_agent import ExecutionAgent
from irs_data_agent.api.irs_client import IRSClient

__all__ = [
    "Workflow",
    "State",
    "PlanningAgent",
    "ExecutionAgent",
    "IRSClient",
]
