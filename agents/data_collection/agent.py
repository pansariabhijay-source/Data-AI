"""
Data Collection Agent — CrewAI agent definition.
"""

from __future__ import annotations

from crewai import Agent, LLM

from agents.base import create_agent
from agents.data_collection.tools import collect_data


def create_data_collection_agent(llm: LLM) -> Agent:
    """Create the data collection agent."""
    return create_agent(
        role="Senior Data Engineer",
        goal=(
            "Ingest the dataset, validate its structure, detect the ML problem type, "
            "profile all columns, check for data quality issues and target leakage, "
            "and prepare the dataset metadata for downstream agents."
        ),
        backstory=(
            "You are a senior data engineer at a top-tier tech company with 10+ years of experience "
            "in data pipelines. You are meticulous about data quality and never let bad data pass "
            "through your pipelines. You profile every dataset thoroughly and flag any anomalies."
        ),
        tools=[collect_data],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
    )
