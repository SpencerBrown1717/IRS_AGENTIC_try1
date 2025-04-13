irs_data_agent/
├── setup.py                    # Simple package setup
├── requirements.txt            # Minimal dependencies
├── cli.py                      # Basic CLI interface
├── irs_data_agent/
    ├── __init__.py
    ├── workflow.py             # Simple workflow engine
    ├── state.py                # Basic state definition
    ├── agents/
    │   ├── __init__.py
    │   ├── planning_agent.py   # Basic planning
    │   └── execution_agent.py  # Combined execution
    ├── api/
    │   ├── __init__.py
    │   └── irs_client.py       # Simple IRS API client
    └── utils/
        ├── __init__.py
        └── formatting.py       # Output formattin