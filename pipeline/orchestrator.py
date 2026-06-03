"""
Pipeline Orchestrator — CrewAI Crew assembly and execution.

Assembles all agents, creates ordered tasks, initializes shared state,
and runs the pipeline with checkpoint support and retry logic.
"""

from __future__ import annotations

import json
from typing import Optional

from crewai import Crew, Process, Task

from agents.base import create_llm
from agents.data_collection.agent import create_data_collection_agent
from agents.data_collection.tools import init_data_collection
from agents.error_detection.agent import create_error_detection_agent
from agents.error_detection.tools import init_error_detection
from agents.feature_engineering.agent import create_feature_engineering_agent
from agents.feature_engineering.tools import init_feature_engineering
from agents.finalization.agent import create_finalization_agent
from agents.finalization.tools import init_finalization
from agents.improvement.agent import create_improvement_agent
from agents.improvement.tools import init_improvement
from agents.manager.agent import create_manager_agent
from agents.preprocessing.agent import create_preprocessing_agent
from agents.preprocessing.tools import init_preprocessing
from agents.splitting.agent import create_splitting_agent
from agents.splitting.tools import init_splitting
from agents.training.agent import create_training_agent
from agents.training.tools import init_training
from core.config import Settings
from core.logging_config import get_logger, set_correlation_id
from core.state import PipelineState
from core.utils import generate_run_id, get_timestamp
from pipeline.checkpoint import CheckpointManager

logger = get_logger("orchestrator")


class PipelineOrchestrator:
    """Assembles and executes the full CrewAI pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._checkpoint = CheckpointManager(
            settings.pipeline.artifact_dir,
            settings.pipeline.checkpoint_enabled,
        )

    def run(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: bool = False,
    ) -> PipelineState:
        """Execute the full pipeline.

        Args:
            data_path: Path to the input dataset.
            target_column: Name of the target column (None for clustering).
            run_id: Optional run ID (generated if not provided).
            resume: Whether to resume from a checkpoint.

        Returns:
            Final PipelineState.
        """
        # Initialize state
        if run_id is None:
            run_id = generate_run_id()

        set_correlation_id(run_id)

        if resume and run_id:
            state = self._checkpoint.load_latest(run_id)
            if state:
                logger.info(f"Resuming from checkpoint, stage: {state.current_stage}")
            else:
                state = self._create_initial_state(run_id, data_path, target_column)
        else:
            state = self._create_initial_state(run_id, data_path, target_column)

        logger.info(f"Pipeline starting: run_id={run_id}, data={data_path}, target={target_column}")

        # Create LLM
        llm = create_llm(self._settings.llm)

        # Initialize all tool services with shared state
        self._init_services(state)

        # Create agents
        dc_agent = create_data_collection_agent(llm)
        pp_agent = create_preprocessing_agent(llm)
        fe_agent = create_feature_engineering_agent(llm)
        sp_agent = create_splitting_agent(llm)
        tr_agent = create_training_agent(llm)
        ed_agent = create_error_detection_agent(llm)
        im_agent = create_improvement_agent(llm)
        fn_agent = create_finalization_agent(llm)
        manager = create_manager_agent(llm)

        # Create tasks (sequential)
        tasks = [
            Task(
                description="Load the dataset, profile all columns, detect the ML problem type, and validate data quality.",
                expected_output="JSON with problem_type, row/column counts, quality score, and any issues found.",
                agent=dc_agent,
            ),
            Task(
                description="Clean the dataset by handling missing values, removing duplicates, fixing dtypes, and winsorizing outliers.",
                expected_output="JSON with before/after row and column counts, quality score, and preprocessing summary.",
                agent=pp_agent,
            ),
            Task(
                description="Engineer features: encode categoricals, extract datetime features, remove low-variance and correlated features, select top features, and scale numerics.",
                expected_output="JSON with feature counts before/after and engineering summary.",
                agent=fe_agent,
            ),
            Task(
                description="Split the dataset into train, validation, and test sets with proper stratification.",
                expected_output="JSON with split paths and ratios.",
                agent=sp_agent,
            ),
            Task(
                description="Train all available models for the detected problem type, evaluate on validation set, and select the best model.",
                expected_output="JSON with all model results, best model name, and primary metric value.",
                agent=tr_agent,
            ),
            Task(
                description="Audit the pipeline for issues: check for low performance, overfitting, data leakage, class imbalance, and other anomalies.",
                expected_output="JSON with all detected issues, severities, and whether retry is recommended.",
                agent=ed_agent,
            ),
            Task(
                description="If errors were detected, attempt to improve performance through hyperparameter tuning.",
                expected_output="JSON with improvement results, retry count, and updated best metric.",
                agent=im_agent,
            ),
            Task(
                description="Save all artifacts (model, metrics, feature importances), generate SHAP explanations, and create the final markdown report.",
                expected_output="JSON with all saved artifact paths and report locations.",
                agent=fn_agent,
            ),
        ]

        # Assemble crew
        crew = Crew(
            agents=[dc_agent, pp_agent, fe_agent, sp_agent, tr_agent, ed_agent, im_agent, fn_agent],
            tasks=tasks,
            process=Process.sequential,
            manager_agent=manager,
            verbose=True,
        )

        # Execute
        try:
            result = crew.kickoff()
            state.completed_at = get_timestamp()
            state.failed = False
            self._checkpoint.save(state, "final")
            logger.info(f"Pipeline completed successfully: {run_id}")
        except Exception as e:
            state.failed = True
            state.failure_reason = str(e)
            state.completed_at = get_timestamp()
            self._checkpoint.save(state, "failed")
            logger.exception(f"Pipeline failed: {e}")

        return state

    def _create_initial_state(self, run_id: str, data_path: str, target: Optional[str]) -> PipelineState:
        return PipelineState(
            run_id=run_id,
            raw_data_path=data_path,
            target_column=target,
            max_retries=self._settings.pipeline.max_retries,
            started_at=get_timestamp(),
        )

    def _init_services(self, state: PipelineState) -> None:
        """Initialize all tool services with the shared state object."""
        s = self._settings
        init_data_collection(state, s)
        init_preprocessing(state, s)
        init_feature_engineering(state, s)
        init_splitting(state, s)
        init_training(state, s)
        init_error_detection(state, s)
        init_improvement(state, s)
        init_finalization(state, s)
