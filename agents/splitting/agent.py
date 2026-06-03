"""Splitting Agent — CrewAI agent definition."""

from __future__ import annotations
from crewai import Agent, LLM
from agents.base import create_agent
from agents.splitting.tools import split_data


def create_splitting_agent(llm: LLM) -> Agent:
    return create_agent(
        role="Data Splitting Specialist",
        goal="Split the dataset into train, validation, and test sets with proper stratification and reproducibility.",
        backstory=(
            "You are an ML engineer who understands the critical importance of proper data splitting. "
            "You prevent data leakage, ensure stratification for classification, and guarantee reproducibility."
        ),
        tools=[split_data],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
    )
