"""
State management for IRS data agent workflows.
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Plan(BaseModel):
    """Represents a plan of action for retrieving and processing IRS data."""
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_step(self, action: str, params: Dict[str, Any] = None):
        """Add a step to the plan."""
        if params is None:
            params = {}
        
        self.steps.append({
            "action": action,
            "params": params,
            "status": "pending",
        })
    
    def mark_step_complete(self, step_index: int, result: Any = None):
        """Mark a step as completed with optional result."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "completed"
            self.steps[step_index]["result"] = result
            self.steps[step_index]["completed_at"] = datetime.now()

class State(BaseModel):
    """
    State management for the IRS data agent.
    Tracks query parameters, results, and processing state.
    """
    # Query parameters
    query: str
    year: Optional[int] = None
    form: Optional[str] = None
    
    # Results storage
    results: List[Dict[str, Any]] = []
    
    # Processing state
    processing_complete: bool = False
    error: Optional[str] = None
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a result to the state."""
        self.results.append(result)
    
    def set_error(self, error_message: str) -> None:
        """Set an error message in the state."""
        self.error = error_message
        
    def mark_complete(self) -> None:
        """Mark processing as complete."""
        self.processing_complete = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "query": self.query,
            "year": self.year,
            "form": self.form,
            "results": self.results,
            "processing_complete": self.processing_complete,
            "error": self.error
        }
