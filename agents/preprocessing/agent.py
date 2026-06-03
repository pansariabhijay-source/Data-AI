"""Preprocessing Agent — CrewAI agent definition."""

from __future__ import annotations

from crewai import Agent, LLM
from agents.base import create_agent
from agents.preprocessing.tools import preprocess_data


def create_preprocessing_agent(llm: LLM) -> Agent:
    return create_agent(
        role="Data Quality Engineer",
        goal=(
            "Clean and preprocess the dataset by handling missing values, removing duplicates, "
            "fixing data types, detecting and winsorizing outliers, and computing data quality scores."
        ),
        backstory=(
            "You are a data quality engineer who has cleaned thousands of production datasets. "
            "You understand that bad data in means bad models out, so you apply rigorous cleaning "
            "while being careful not to lose valuable signal."
        ),
        tools=[preprocess_data],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
    )
