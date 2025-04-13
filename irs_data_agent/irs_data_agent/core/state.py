"""
Enhanced state management for IRS data agent workflows.
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json

class Plan(BaseModel):
    """Represents a plan of action for retrieving and processing IRS data."""
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[Dict[str, Any]] = None
    
    def add_step(self, action: str, params: Dict[str, Any] = None, description: str = None):
        """Add a step to the plan."""
        if params is None:
            params = {}
        
        step = {
            "action": action,
            "params": params,
            "status": "pending",
            "id": len(self.steps),
        }
        
        if description:
            step["description"] = description
            
        self.steps.append(step)
        self.updated_at = datetime.now()
        return step["id"]
    
    def mark_step_running(self, step_index: int):
        """Mark a step as running."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "running"
            self.steps[step_index]["started_at"] = datetime.now()
            self.updated_at = datetime.now()
    
    def mark_step_complete(self, step_index: int, result: Any = None):
        """Mark a step as completed with optional result."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "completed"
            self.steps[step_index]["completed_at"] = datetime.now()
            
            if result is not None:
                self.steps[step_index]["result"] = result
                
            self.updated_at = datetime.now()
            
    def mark_step_failed(self, step_index: int, error: Any = None):
        """Mark a step as failed with optional error."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "failed"
            self.steps[step_index]["failed_at"] = datetime.now()
            
            if error is not None:
                if isinstance(error, Exception):
                    self.steps[step_index]["error"] = str(error)
                else:
                    self.steps[step_index]["error"] = error
                    
            self.updated_at = datetime.now()
    
    def get_step_status(self, step_index: int) -> str:
        """Get the status of a step."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]["status"]
        return "unknown"
    
    def has_failed_steps(self) -> bool:
        """Check if the plan has any failed steps."""
        return any(step["status"] == "failed" for step in self.steps)
    
    def get_failed_steps(self) -> List[Dict[str, Any]]:
        """Get all failed steps."""
        return [step for step in self.steps if step["status"] == "failed"]
    
    def get_pending_steps(self) -> List[Dict[str, Any]]:
        """Get all pending steps."""
        return [step for step in self.steps if step["status"] == "pending"]
    
    def get_running_steps(self) -> List[Dict[str, Any]]:
        """Get all running steps."""
        return [step for step in self.steps if step["status"] == "running"]
    
    def get_completed_steps(self) -> List[Dict[str, Any]]:
        """Get all completed steps."""
        return [step for step in self.steps if step["status"] == "completed"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return json.loads(self.json())


class FormReference(BaseModel):
    """Reference to an IRS form."""
    form_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    relevance_score: Optional[float] = None
    
    @validator('relevance_score')
    def check_relevance_score(cls, v):
        """Validate relevance score is between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Relevance score must be between 0 and 1')
        return v


class AnalysisResult(BaseModel):
    """Result of a query analysis."""
    possible_form_types: List[str] = Field(default_factory=list)
    possible_years: List[int] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None


class State(BaseModel):
    """
    Enhanced state management for IRS data retrieval workflow.
    """
    query: str
    year: Optional[int] = None
    form: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    current_plan: Optional[Plan] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    is_batch_mode: bool = False
    batch_id: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[Dict[str, Any]] = None
    
    # Enhanced state tracking
    query_analysis: Optional[AnalysisResult] = None
    form_references: List[FormReference] = Field(default_factory=list)
    retrieved_data: Dict[str, Any] = Field(default_factory=dict)
    processed_data: Dict[str, Any] = Field(default_factory=dict)
    execution_stats: Dict[str, Any] = Field(default_factory=dict)
    
    def update_with_results(self, new_results: List[Dict[str, Any]]):
        """
        Update state with new results.
        
        Args:
            new_results: New results to add
        """
        self.results.extend(new_results)
        self.metadata["last_updated"] = datetime.now()
        self.updated_at = datetime.now()
        
        # Update execution stats
        self.execution_stats["total_results"] = len(self.results)
        self.execution_stats["last_batch_size"] = len(new_results)
        
    def update_with_analysis(self, analysis: Union[AnalysisResult, Dict[str, Any]]):
        """
        Update state with query analysis.
        
        Args:
            analysis: Analysis result to add
        """
        if isinstance(analysis, dict):
            self.query_analysis = AnalysisResult(**analysis)
        else:
            self.query_analysis = analysis
            
        self.updated_at = datetime.now()
        
    def add_form_reference(self, form_reference: Union[FormReference, Dict[str, Any]]):
        """
        Add a form reference to the state.
        
        Args:
            form_reference: Form reference to add
        """
        if isinstance(form_reference, dict):
            self.form_references.append(FormReference(**form_reference))
        else:
            self.form_references.append(form_reference)
            
        self.updated_at = datetime.now()
        
    def add_form_references(self, form_references: List[Union[FormReference, Dict[str, Any]]]):
        """
        Add multiple form references to the state.
        
        Args:
            form_references: Form references to add
        """
        for ref in form_references:
            self.add_form_reference(ref)
            
    def set_status(self, status: str, error: Any = None):
        """
        Set the state status and optional error.
        
        Args:
            status: New status (pending, running, completed, failed)
            error: Optional error information
        """
        self.status = status
        
        if error is not None:
            if isinstance(error, Exception):
                self.error = {"message": str(error), "type": type(error).__name__}
            else:
                self.error = error
                
        self.updated_at = datetime.now()
        
    def is_complete(self) -> bool:
        """
        Check if the workflow is complete.
        
        Returns:
            Boolean indicating completion status
        """
        # Basic implementation - workflow is complete if status is completed or failed
        return self.status in ["completed", "failed"]
    
    def is_successful(self) -> bool:
        """
        Check if the workflow completed successfully.
        
        Returns:
            Boolean indicating success status
        """
        return self.status == "completed" and len(self.results) > 0
    
    def reset(self):
        """Reset the mutable parts of the state."""
        self.results = []
        self.current_plan = None
        self.metadata = {}
        self.updated_at = datetime.now()
        self.status = "pending"
        self.error = None
        self.query_analysis = None
        self.form_references = []
        self.retrieved_data = {}
        self.processed_data = {}
        self.execution_stats = {}
        
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state.
        
        Returns:
            Dict containing state summary
        """
        return {
            "query": self.query,
            "form": self.form,
            "year": self.year,
            "status": self.status,
            "result_count": len(self.results),
            "has_error": self.error is not None,
            "has_plan": self.current_plan is not None,
            "step_count": len(self.current_plan.steps) if self.current_plan else 0,
            "form_references_count": len(self.form_references),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_stats": self.execution_stats
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return json.loads(self.json())
