#!/usr/bin/env python3
import typer
from rich.console import Console
from rich import print as rprint
from irs_data_agent.api.irs_client import IRSClient

app = typer.Typer(help="IRS Data Agent CLI")
console = Console()

@app.command()
def search(
    term: str = typer.Argument(..., help="Search term for IRS data"),
    year: int = typer.Option(None, help="Tax year to search"),
    limit: int = typer.Option(10, help="Maximum number of results to return")
):
    """Search IRS data sources for specific information."""
    console.print(f"[bold green]Searching for:[/] {term}")
    if year:
        console.print(f"[bold blue]Year:[/] {year}")
    console.print(f"[bold yellow]Limit:[/] {limit}")
    
    # Placeholder for actual search implementation
    console.print("[italic]Search functionality will be implemented here[/]")

@app.command()
def download(
    form_number: str = typer.Argument(..., help="IRS form number to download"),
    year: int = typer.Option(None, help="Tax year for the form"),
    output_dir: str = typer.Option("./downloads", help="Directory to save downloaded files")
):
    """Download IRS forms and publications."""
    console.print(f"[bold green]Downloading form:[/] {form_number}")
    if year:
        console.print(f"[bold blue]Year:[/] {year}")
    console.print(f"[bold yellow]Output directory:[/] {output_dir}")
    
    # Placeholder for actual download implementation
    console.print("[italic]Download functionality will be implemented here[/]")

@app.command()
def info():
    """Display information about the IRS Data Agent."""
    console.print("[bold]IRS Data Agent[/]", style="green")
    console.print("Version: 0.1.0")
    console.print("A toolkit for interacting with IRS data sources")
    console.print("\n[bold]Available Commands:[/]")
    console.print("  search    - Search IRS data sources")
    console.print("  download  - Download IRS forms and publications")
    console.print("  info      - Display this information")

@app.command()
def forms(
    form_type: str = typer.Argument(None, help="Filter by form type"),
):
    """List available IRS forms."""
    try:
        console.print("[bold blue]Retrieving available IRS forms...[/bold blue]")
        
        client = IRSClient()
        forms = client.list_forms(form_type=form_type)
        
        console.print("[bold green]Available Forms:[/bold green]")
        for form in forms:
            console.print(f"- {form['form_number']}: {form['title']}")
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

def main():
    """Entry point for the CLI application."""
    app()

if __name__ == "__main__":
    main()
