#!/usr/bin/env python3
"""
Enhanced CLI interface for IRS Data Agent.
"""
import os
import sys
import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from irs_data_agent.core.workflow import Workflow
from irs_data_agent.core.state import State
from irs_data_agent.agents.planning_agent import PlanningAgent
from irs_data_agent.agents.retrieval_agent import RetrievalAgent
from irs_data_agent.agents.processing_agent import ProcessingAgent
from irs_data_agent.agents.error_agent import ErrorAgent
from irs_data_agent.api.irs_client import IRSClient
from irs_data_agent.utils.logging import setup_logging, get_logger
from irs_data_agent.utils.formatting import format_results
from irs_data_agent.data.cache_manager import CacheManager

app = typer.Typer(help="IRS Data Agent CLI", add_completion=False)
console = Console()
logger = get_logger(__name__)

# Initialize app config
try:
    from irs_data_agent.utils.config import load_config
    config = load_config()
except Exception as e:
    console.print(f"[bold red]Error loading configuration: {str(e)}[/bold red]")
    config = {}

# Create app directories
for directory in ['logs', '.cache']:
    os.makedirs(directory, exist_ok=True)

# Setup logging
setup_logging()


def create_workflow(query: str, year: Optional[int] = None, form: Optional[str] = None,
                   use_cache: bool = True) -> Workflow:
    """Helper function to create and configure a workflow."""
    # Initialize components
    state = State(query=query, year=year, form=form)
    client = IRSClient()
    
    # Initialize agents
    planning_agent = PlanningAgent()
    retrieval_agent = RetrievalAgent(client=client)
    processing_agent = ProcessingAgent()
    error_agent = ErrorAgent()
    
    # Initialize cache if enabled
    cache = None
    if use_cache and config.get('cache', {}).get('enabled', True):
        cache = CacheManager()
    
    # Set up workflow
    workflow = Workflow(
        state=state,
        planning_agent=planning_agent,
        retrieval_agent=retrieval_agent,
        processing_agent=processing_agent,
        error_agent=error_agent,
        cache=cache
    )
    
    return workflow


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query for IRS data"),
    year: Optional[int] = typer.Option(None, help="Tax year to search"),
    form: Optional[str] = typer.Option(None, help="IRS form type"),
    output: str = typer.Option("text", help="Output format (text, json, csv)"),
    save: Optional[Path] = typer.Option(None, help="Save results to file"),
    no_cache: bool = typer.Option(False, help="Disable cache usage"),
    verbose: bool = typer.Option(False, help="Enable verbose output")
):
    """Search IRS data with the specified parameters."""
    try:
        logger.info(f"Starting search with query: {query}")
        
        if verbose:
            console.print(f"[bold blue]Searching IRS data with query: {query}[/bold blue]")
            if year:
                console.print(f"[blue]Year: {year}[/blue]")
            if form:
                console.print(f"[blue]Form: {form}[/blue]")
        
        # Create and run workflow with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[green]Searching...", total=None)
            
            workflow = create_workflow(query, year, form, not no_cache)
            results = workflow.run()
            
            progress.update(task, completed=True, description="[green]Search completed!")
        
        # Format and display results
        formatted_results = format_results(results, output_format=output)
        console.print(formatted_results)
        
        # Save results if requested
        if save:
            with open(save, "w") as f:
                f.write(formatted_results)
            console.print(f"[bold green]Results saved to {save}[/bold green]")
        
        logger.info(f"Search completed successfully with {results.get('total_results', 0)} results")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Search canceled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def forms(
    form_type: Optional[str] = typer.Argument(None, help="Filter by form type"),
    year: Optional[int] = typer.Option(None, help="Filter by tax year"),
    output: str = typer.Option("text", help="Output format (text, json, csv)"),
    save: Optional[Path] = typer.Option(None, help="Save results to file")
):
    """List available IRS forms."""
    try:
        logger.info(f"Listing forms with type: {form_type}, year: {year}")
        console.print("[bold blue]Retrieving available IRS forms...[/bold blue]")
        
        client = IRSClient()
        forms = client.list_forms(form_type=form_type, year=year)
        
        # Format and display results
        result_dict = {
            "query": f"Forms {form_type or 'all'} {year or 'all years'}",
            "total_results": len(forms),
            "results": forms
        }
        
        formatted_results = format_results(result_dict, output_format=output)
        console.print(formatted_results)
        
        # Save results if requested
        if save:
            with open(save, "w") as f:
                f.write(formatted_results)
            console.print(f"[bold green]Results saved to {save}[/bold green]")
            
        logger.info(f"Forms listing completed with {len(forms)} results")
            
    except Exception as e:
        logger.error(f"Error listing forms: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def batch(
    file: Path = typer.Argument(..., help="File with search queries (one per line)"),
    year: Optional[int] = typer.Option(None, help="Tax year to search"),
    output: str = typer.Option("json", help="Output format (text, json, csv)"),
    save_dir: Path = typer.Option("./results", help="Directory to save results"),
    parallel: int = typer.Option(2, help="Number of parallel searches"),
    no_cache: bool = typer.Option(False, help="Disable cache usage"),
):
    """Run batch processing of multiple searches from a file."""
    try:
        from irs_data_agent.data.batch_processor import BatchProcessor
        
        logger.info(f"Starting batch processing from file: {file}")
        console.print(f"[bold blue]Starting batch processing from file: {file}[/bold blue]")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Read queries from file
        with open(file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
            
        console.print(f"[blue]Found {len(queries)} queries to process[/blue]")
        
        # Confirm before proceeding with large batch
        if len(queries) > 10:
            if not Confirm.ask(f"Process {len(queries)} queries? This might take a while"):
                console.print("[yellow]Batch processing canceled[/yellow]")
                return
        
        # Create batch processor and run
        processor = BatchProcessor(
            queries=queries,
            year=year,
            output_format=output,
            save_dir=save_dir,
            workers=parallel,
            use_cache=not no_cache
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[green]Processing batch...", total=len(queries))
            
            def update_progress(completed, total, current_query):
                progress.update(task, completed=completed, description=f"[green]Processing {completed}/{total}: {current_query}")
            
            results = processor.process(progress_callback=update_progress)
            
        console.print(f"[bold green]Batch processing completed. Results saved to {save_dir}[/bold green]")
        console.print(f"Processed {len(results)} queries with {sum(1 for r in results if r.get('success', False))} successes")
        
        logger.info(f"Batch processing completed with {len(results)} results")
        
    except FileNotFoundError:
        console.print(f"[bold red]Error: File not found: {file}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def status():
    """Show system status and cache information."""
    try:
        logger.info("Showing system status")
        console.print("[bold blue]IRS Data Agent Status[/bold blue]")
        
        # Display configuration
        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"API Base URL: {config.get('api', {}).get('base_url', 'Not configured')}")
        console.print(f"Cache Enabled: {config.get('cache', {}).get('enabled', False)}")
        console.print(f"Log Level: {config.get('logging', {}).get('level', 'INFO')}")
        
        # Display cache information if enabled
        if config.get('cache', {}).get('enabled', False):
            cache = CacheManager()
            cache_info = cache.get_stats()
            
            console.print("\n[bold]Cache Information:[/bold]")
            console.print(f"Cache Size: {cache_info['size_mb']:.2f} MB")
            console.print(f"Items in Cache: {cache_info['item_count']}")
            console.print(f"Hit Rate: {cache_info['hit_rate']:.2f}%")
            
            # Ask if user wants to clear cache
            if cache_info['item_count'] > 0:
                if Confirm.ask("Would you like to clear the cache?"):
                    cache.clear()
                    console.print("[green]Cache cleared successfully[/green]")
        
        # Display API status
        console.print("\n[bold]API Status:[/bold]")
        client = IRSClient()
        api_status = client.get_status()
        
        if api_status.get('status') == 'operational':
            console.print(f"[green]API Status: {api_status.get('status', 'Unknown')}[/green]")
        else:
            console.print(f"[yellow]API Status: {api_status.get('status', 'Unknown')}[/yellow]")
            
        if 'message' in api_status:
            console.print(f"Message: {api_status['message']}")
            
        logger.info("Status command completed successfully")
        
    except Exception as e:
        logger.error(f"Error showing status: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def config_show():
    """Show current configuration."""
    try:
        import yaml
        
        logger.info("Showing configuration")
        console.print("[bold blue]Current Configuration[/bold blue]\n")
        
        # Pretty print the config
        yaml_config = yaml.dump(config, sort_keys=False, default_flow_style=False)
        console.print(yaml_config)
        
    except Exception as e:
        logger.error(f"Error showing configuration: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


def main():
    """Main entry point for the CLI."""
    # Initialize environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    app()


if __name__ == "__main__":
    main()