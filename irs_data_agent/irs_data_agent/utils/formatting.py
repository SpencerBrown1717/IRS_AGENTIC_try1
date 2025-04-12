"""
Enhanced formatting utilities for IRS data agent output.
"""
from typing import Dict, List, Any, Optional
import json
import csv
import yaml
from io import StringIO
from datetime import datetime, date
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.columns import Columns
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich import box

from irs_data_agent.utils.logging import get_logger

logger = get_logger(__name__)

# Custom JSON encoder to handle dates and other complex types
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle dates and other complex types."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        # Handle other special types
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # Convert any unserializable objects to strings

def format_results(results: Dict[str, Any], output_format: str = "text") -> str:
    """
    Format results in the specified output format.
    
    Args:
        results: Results to format
        output_format: Output format (text, json, csv, yaml)
        
    Returns:
        Formatted results as a string
    """
    logger.debug(f"Formatting results in {output_format} format")
    
    if not isinstance(results, dict):
        logger.warning(f"Expected dict for results, got {type(results).__name__}")
        results = {"error": f"Invalid results type: {type(results).__name__}, expected dict"}
    
    output_format = output_format.lower()
    if output_format == "json":
        return format_json(results)
    elif output_format == "csv":
        return format_csv(results)
    elif output_format == "yaml":
        return format_yaml(results)
    else:  # Default to text
        return format_text(results)

def format_json(results: Dict[str, Any], indent: int = 2) -> str:
    """
    Format results as JSON.
    
    Args:
        results: Results to format
        indent: JSON indentation level
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(results, indent=indent, cls=CustomJSONEncoder)
    except Exception as e:
        logger.error(f"Error formatting JSON: {str(e)}")
        return json.dumps({"error": f"Failed to format results as JSON: {str(e)}"})

def format_csv(results: Dict[str, Any]) -> str:
    """
    Format results as CSV.
    
    Args:
        results: Results to format
        
    Returns:
        CSV string
    """
    output = StringIO()
    writer = None
    
    try:
        # Extract results data
        result_items = []
        
        if "results" in results:
            result_items = results["results"]
        elif "results_by_type" in results:
            # Flatten results from different types
            for form_type, items in results["results_by_type"].items():
                for item in items:
                    # Add form_type if not already present
                    if "form_type" not in item:
                        item["form_type"] = form_type
                    result_items.append(item)
        
        if not result_items:
            # No results to format
            writer = csv.writer(output)
            writer.writerow(["No results found"])
            return output.getvalue()
        
        # Determine all possible fields
        all_fields = set()
        for item in result_items:
            all_fields.update(item.keys())
        
        # Prioritize common fields at the beginning
        priority_fields = ["form_id", "title", "form_type", "year", "url", "relevance_score"]
        field_order = [f for f in priority_fields if f in all_fields]
        field_order.extend([f for f in all_fields if f not in priority_fields])
        
        # Write header row
        writer = csv.writer(output)
        writer.writerow(field_order)
        
        # Write data rows
        for item in result_items:
            row = []
            for field in field_order:
                value = item.get(field, "")
                # Convert non-string values to strings
                if not isinstance(value, str):
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, cls=CustomJSONEncoder)
                    else:
                        value = str(value)
                row.append(value)
            writer.writerow(row)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error formatting CSV: {str(e)}")
        if writer:
            writer.writerow(["Error", f"Failed to format results as CSV: {str(e)}"])
        else:
            output.write(f"Error: Failed to format results as CSV: {str(e)}")
        return output.getvalue()

def format_yaml(results: Dict[str, Any]) -> str:
    """
    Format results as YAML.
    
    Args:
        results: Results to format
        
    Returns:
        YAML string
    """
    try:
        return yaml.dump(results, sort_keys=False, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error formatting YAML: {str(e)}")
        return f"Error: Failed to format results as YAML: {str(e)}"

def format_text(results: Dict[str, Any]) -> str:
    """
    Format results as rich text for console display.
    
    Args:
        results: Results to format
        
    Returns:
        Formatted output as a string
    """
    console = Console(record=True)
    
    try:
        # Display query and metadata
        query = results.get('query', '')
        year = results.get('year')
        form = results.get('form')
        
        title = f"Query: {query}"
        if year:
            title += f", Year: {year}"
        if form:
            title += f", Form: {form}"
            
        console.print(Panel(title, title="IRS Data Agent Results"))
        
        # Display stats
        total_results = results.get('total_results', 0)
        status = results.get('status', 'completed')
        form_types = results.get('form_types', [])
        
        stats = [
            f"[bold]Total Results:[/bold] {total_results}",
            f"[bold]Status:[/bold] {status}"
        ]
        
        if form_types:
            stats.append(f"[bold]Form Types:[/bold] {', '.join(form_types)}")
            
        console.print(Columns(stats))
        console.print()
        
        # Check for errors
        if 'error' in results:
            error = results['error']
            error_msg = error if isinstance(error, str) else error.get('message', str(error))
            console.print(Panel(f"[bold red]{error_msg}[/bold red]", title="Error"))
            console.print()
        
        # Display results
        if "results_by_type" in results:
            _print_results_by_type(console, results["results_by_type"])
        elif "results" in results:
            _print_results_list(console, results["results"])
        
        # Display execution stats if present
        if "stats" in results:
            _print_execution_stats(console, results["stats"])
            
        # Return the captured output
        return console.export_text()
        
    except Exception as e:
        logger.error(f"Error formatting text: {str(e)}")
        console.print(f"[bold red]Error formatting results: {str(e)}[/bold red]")
        return console.export_text()

def _print_results_by_type(console: Console, results_by_type: Dict[str, List[Dict[str, Any]]]):
    """
    Print results grouped by type.
    
    Args:
        console: Rich console
        results_by_type: Results grouped by type
    """
    if not isinstance(results_by_type, dict):
        console.print(f"[bold red]Invalid results_by_type: expected dict, got {type(results_by_type).__name__}[/bold red]")
        return
        
    for form_type, forms in results_by_type.items():
        if not forms:
            continue
            
        console.print(f"[bold]{form_type.capitalize()} Forms:[/bold]")
        
        table = Table(show_header=True, header_style="bold", box=box.ROUNDED)
        table.add_column("Form ID")
        table.add_column("Title")
        table.add_column("Year")
        
        # Add relevance column if present in any result
        has_relevance = any("relevance_score" in form for form in forms)
        if has_relevance:
            table.add_column("Relevance")
        
        # Add URL column if present in any result
        has_url = any("url" in form for form in forms)
        if has_url:
            table.add_column("URL")
        
        for form in forms:
            row = [
                str(form.get("form_id", "")),
                str(form.get("title", "")),
                str(form.get("year", ""))
            ]
            
            if has_relevance:
                score = form.get("relevance_score")
                if score is not None:
                    try:
                        # Format as percentage with color
                        score_float = float(score)
                        score_str = f"{score_float:.0%}"
                        if score_float >= 0.8:
                            row.append(f"[green]{score_str}[/green]")
                        elif score_float >= 0.5:
                            row.append(f"[yellow]{score_str}[/yellow]")
                        else:
                            row.append(f"[red]{score_str}[/red]")
                    except (ValueError, TypeError):
                        row.append(str(score))
                else:
                    row.append("")
            
            if has_url:
                row.append(str(form.get("url", "")))
                
            table.add_row(*row)
            
        console.print(table)
        console.print()

def _print_results_list(console: Console, results: List[Dict[str, Any]]):
    """
    Print a flat list of results.
    
    Args:
        console: Rich console
        results: Results list
    """
    if not isinstance(results, list):
        console.print(f"[bold red]Invalid results: expected list, got {type(results).__name__}[/bold red]")
        return
        
    if not results:
        console.print("[italic]No results found[/italic]")
        return
        
    table = Table(show_header=True, header_style="bold", box=box.ROUNDED)
    table.add_column("Form ID")
    table.add_column("Title")
    table.add_column("Type")
    table.add_column("Year")
    
    # Add relevance column if present in any result
    has_relevance = any("relevance_score" in result for result in results)
    if has_relevance:
        table.add_column("Relevance")
    
    # Add URL column if present in any result
    has_url = any("url" in result for result in results)
    if has_url:
        table.add_column("URL")
    
    for result in results:
        row = [
            str(result.get("form_id", "")),
            str(result.get("title", "")),
            str(result.get("form_type", "")),
            str(result.get("year", ""))
        ]
        
        if has_relevance:
            score = result.get("relevance_score")
            if score is not None:
                try:
                    # Format as percentage with color
                    score_float = float(score)
                    score_str = f"{score_float:.0%}"
                    if score_float >= 0.8:
                        row.append(f"[green]{score_str}[/green]")
                    elif score_float >= 0.5:
                        row.append(f"[yellow]{score_str}[/yellow]")
                    else:
                        row.append(f"[red]{score_str}[/red]")
                except (ValueError, TypeError):
                    row.append(str(score))
            else:
                row.append("")
        
        if has_url:
            row.append(str(result.get("url", "")))
            
        table.add_row(*row)
        
    console.print(table)
    console.print()

def _print_execution_stats(console: Console, stats: Dict[str, Any]):
    """
    Print execution statistics.
    
    Args:
        console: Rich console
        stats: Execution statistics
    """
    if not isinstance(stats, dict):
        console.print(f"[bold red]Invalid stats: expected dict, got {type(stats).__name__}[/bold red]")
        return
        
    console.print("[bold]Execution Statistics:[/bold]")
    
    # Format stats into a list of key-value pairs
    stat_items = []
    for key, value in stats.items():
        # Skip complex nested structures
        if isinstance(value, (dict, list)):
            continue
            
        # Format key for display
        display_key = key.replace("_", " ").title()
        
        # Format value based on type
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        elif isinstance(value, (datetime, date)):
            display_value = value.isoformat()
        else:
            display_value = str(value)
            
        stat_items.append(f"[bold]{display_key}:[/bold] {display_value}")
    
    # Display in columns
    console.print(Columns(stat_items, equal=True, expand=True))
    console.print()

def format_plan(plan: Dict[str, Any]) -> str:
    """
    Format a plan for display.
    
    Args:
        plan: Plan to format
        
    Returns:
        Formatted plan as a string
    """
    console = Console(record=True)
    
    try:
        if not isinstance(plan, dict) or "steps" not in plan:
            return f"Invalid plan format: {type(plan).__name__}, expected dict with 'steps' key"
            
        tree = Tree("[bold]Plan[/bold]")
        
        for i, step in enumerate(plan.get("steps", [])):
            status = step.get("status", "pending")
            action = step.get("action", "unknown")
            description = step.get("description", f"Step {i+1}: {action}")
            
            if status == "completed":
                step_node = tree.add(f"[green]{description} (Completed)[/green]")
            elif status == "running":
                step_node = tree.add(f"[yellow]{description} (Running)[/yellow]")
            elif status == "failed":
                step_node = tree.add(f"[red]{description} (Failed)[/red]")
            else:
                step_node = tree.add(f"{description} (Pending)")
                
            # Add parameters
            params = step.get("params", {})
            if params:
                try:
                    params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                    step_node.add(f"[dim]Parameters:[/dim] {params_str}")
                except Exception as e:
                    step_node.add(f"[dim]Parameters:[/dim] Error formatting parameters: {str(e)}")
                
            # Add result if completed
            if status == "completed" and "result" in step:
                result = step["result"]
                try:
                    if isinstance(result, dict):
                        result_str = ", ".join([f"{k}={v}" for k, v in result.items() if not isinstance(v, (dict, list))])
                        step_node.add(f"[dim]Result:[/dim] {result_str}")
                    elif isinstance(result, list):
                        step_node.add(f"[dim]Result:[/dim] {len(result)} items")
                    else:
                        step_node.add(f"[dim]Result:[/dim] {result}")
                except Exception as e:
                    step_node.add(f"[dim]Result:[/dim] Error formatting result: {str(e)}")
                    
            # Add error if failed
            if status == "failed" and "error" in step:
                error = step["error"]
                step_node.add(f"[dim]Error:[/dim] [red]{error}[/red]")
        
        console.print(tree)
        return console.export_text()
        
    except Exception as e:
        logger.error(f"Error formatting plan: {str(e)}")
        console.print(f"[bold red]Error formatting plan: {str(e)}[/bold red]")
        return console.export_text()
