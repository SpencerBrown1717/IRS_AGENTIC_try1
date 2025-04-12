"""
Formatting utilities for IRS data agent output.
"""
from typing import Dict, List, Any
import json
import csv
from io import StringIO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.markdown import Markdown

console = Console()

def format_results(results: Dict[str, Any], output_format: str = "text") -> str:
    """
    Format results in the specified output format.
    
    Args:
        results: Results to format
        output_format: Output format (text, json, csv)
        
    Returns:
        Formatted results as a string
    """
    if output_format == "json":
        return format_json(results)
    elif output_format == "csv":
        return format_csv(results)
    else:  # Default to text
        return format_text(results)

def format_json(results: Dict[str, Any]) -> str:
    """Format results as JSON."""
    return json.dumps(results, indent=2)

def format_csv(results: Dict[str, Any]) -> str:
    """Format results as CSV."""
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header row
    writer.writerow(["form_id", "title", "form_type", "year", "url"])
    
    # Write data rows
    if "results" in results:
        for item in results["results"]:
            writer.writerow([
                item.get("form_id", ""),
                item.get("title", ""),
                item.get("form_type", ""),
                item.get("year", ""),
                item.get("url", "")
            ])
    
    return output.getvalue()

def format_text(results: Dict[str, Any]) -> str:
    """
    Format results as rich text for console display.
    
    Returns the formatted output as a string that can be printed by rich.Console.
    """
    console = Console(record=True)
    
    # Display query and metadata
    console.print(Panel(f"[bold]Query:[/bold] {results.get('query', '')}", 
                       title="IRS Data Agent Results"))
    
    # Display stats
    console.print(f"[bold]Total Results:[/bold] {results.get('total_results', 0)}")
    console.print(f"[bold]Form Types:[/bold] {', '.join(results.get('form_types', []))}")
    console.print()
    
    # Display results by form type
    if "results_by_type" in results:
        for form_type, forms in results["results_by_type"].items():
            console.print(f"[bold]{form_type.capitalize()} Forms:[/bold]")
            
            table = Table(show_header=True, header_style="bold")
            table.add_column("Form ID")
            table.add_column("Title")
            table.add_column("Year")
            table.add_column("URL")
            
            for form in forms:
                table.add_row(
                    form.get("form_id", ""),
                    form.get("title", ""),
                    str(form.get("year", "")),
                    form.get("url", "")
                )
                
            console.print(table)
            console.print()
    
    # Return the captured output
    return console.export_text()

def format_plan(plan: Dict[str, Any]) -> str:
    """
    Format a plan for display.
    
    Args:
        plan: Plan to format
        
    Returns:
        Formatted plan as a string
    """
    console = Console(record=True)
    
    tree = Tree("[bold]Plan[/bold]")
    
    for i, step in enumerate(plan.get("steps", [])):
        status = step.get("status", "pending")
        
        if status == "completed":
            step_node = tree.add(f"[green]Step {i+1}: {step['action']} (Completed)[/green]")
        else:
            step_node = tree.add(f"Step {i+1}: {step['action']} (Pending)")
            
        # Add parameters
        params = step.get("params", {})
        if params:
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            step_node.add(f"[dim]Parameters:[/dim] {params_str}")
            
        # Add result if completed
        if status == "completed" and "result" in step:
            result = step["result"]
            if isinstance(result, dict):
                result_str = ", ".join([f"{k}={v}" for k, v in result.items() if not isinstance(v, (dict, list))])
                step_node.add(f"[dim]Result:[/dim] {result_str}")
            elif isinstance(result, list):
                step_node.add(f"[dim]Result:[/dim] {len(result)} items")
            else:
                step_node.add(f"[dim]Result:[/dim] {result}")
    
    console.print(tree)
    return console.export_text() 

def format_results_list(results: List[Dict[str, Any]]) -> None:
    """
    Format and display search results in a rich table.
    
    Args:
        results: List of result dictionaries from IRS API
    """
    if not results:
        console.print(Panel("[italic]No results found[/]", title="Results", border_style="yellow"))
        return
    
    table = Table(title=f"Found {len(results)} Results")
    
    # Determine columns based on first result
    if results:
        columns = list(results[0].keys())
        # Prioritize certain columns
        priority_columns = ["title", "form_number", "year", "description", "url"]
        for col in priority_columns:
            if col in columns:
                table.add_column(col.replace("_", " ").title(), style="green")
        
        # Add remaining columns
        for col in columns:
            if col not in priority_columns:
                table.add_column(col.replace("_", " ").title())
    
    # Add rows
    for result in results:
        row = []
        for col in table.columns:
            col_name = col.header.lower().replace(" ", "_")
            value = result.get(col_name, "")
            if isinstance(value, (dict, list)):
                value = str(value)
            row.append(str(value))
        table.add_row(*row)
    
    console.print(table)

def format_plan_object(plan: Any) -> None:
    """
    Format and display a retrieval plan.
    
    Args:
        plan: Plan object with steps to execute
    """
    console.print(Panel(f"[bold]{plan.description}[/]", title="Plan", border_style="blue"))
    
    table = Table(title="Plan Steps")
    table.add_column("Step", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")
    
    for i, step in enumerate(plan.steps):
        params_str = ", ".join([f"{k}={v}" for k, v in step.parameters.items()])
        table.add_row(
            str(i+1) + ". " + step.name,
            step.description,
            params_str
        )
    
    console.print(table)

def format_form_info(form: Dict[str, Any]) -> None:
    """
    Format and display information about an IRS form.
    
    Args:
        form: Dictionary containing form information
    """
    if not form:
        console.print(Panel("[italic]Form not found[/]", title="Form Information", border_style="red"))
        return
    
    title = form.get("title", "Unknown Form")
    form_number = form.get("form_number", "Unknown")
    
    panel_title = f"Form {form_number}: {title}"
    
    content = []
    for key, value in form.items():
        if key not in ["title", "form_number"]:
            content.append(f"[bold]{key.replace('_', ' ').title()}:[/] {value}")
    
    content_str = "\n".join(content)
    console.print(Panel(content_str, title=panel_title, border_style="green"))

def format_error(error_message: str) -> None:
    """
    Format and display an error message.
    
    Args:
        error_message: Error message to display
    """
    console.print(Panel(f"[bold red]{error_message}[/]", title="Error", border_style="red"))
