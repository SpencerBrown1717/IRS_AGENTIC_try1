This needed to be updated and will be usfull to to have as an api as a general purupious Finance aplication. 

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



flowchart TD
    CLI[CLI Interface] --> Workflow
    
    subgraph Application
        Workflow[workflow.py] --> State[state.py]
        Workflow --> Planning[planning_agent.py]
        Planning --> State
        Planning --> Execution[execution_agent.py]
        Execution --> IRSClient[irs_client.py]
        Execution --> State
        IRSClient --> Formatting[formatting.py]
        Formatting --> State
    end
    
    IRSClient <--> IRS[IRS API]
    
    State --> Output[Results/Output]
    
    style Application fill:#f5f5f5,stroke:#333,stroke-width:1px
    style IRS fill:#e1f5fe,stroke:#0288d1,stroke-width:1px
    style Output fill:#e8f5e9,stroke:#4caf50,stroke-width:1px
