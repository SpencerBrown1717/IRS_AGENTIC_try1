from setuptools import setup, find_packages

setup(
    name="irs_data_agent",
    version="0.1.0",
    description="Agent-based toolkit for interacting with IRS data",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=1.8.0",
        "typer>=0.3.2",
        "rich>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "irs-agent=cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
