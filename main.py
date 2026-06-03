"""
Autonomous Data Scientist Pipeline — Entry Point.

Usage:
    python main.py --data path/to/data.csv --target target_column
    python main.py --data data/sample_iris.csv --target species
    python main.py --data data/housing.csv --target price --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Main entry point for the autonomous data scientist pipeline."""
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Autonomous Data Scientist Pipeline — CrewAI Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to input dataset (CSV)")
    parser.add_argument("--target", default=None, help="Target column name (omit for clustering)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--run-id", default=None, help="Custom run ID (generated if omitted)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Validate data path
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    # Load settings
    from core.config import load_settings
    from core.logging_config import setup_logging

    settings = load_settings(config_path=args.config)

    if args.verbose:
        settings.logging.level = "DEBUG"

    # Setup logging
    setup_logging(
        log_dir=settings.pipeline.log_dir,
        level=settings.logging.level,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
        json_format=settings.logging.json_format,
    )

    from core.logging_config import get_logger
    logger = get_logger("main")
    logger.info("=" * 60)
    logger.info("Autonomous Data Scientist Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
    logger.info(f"Target: {args.target or '(clustering mode)'}")
    logger.info(f"Config: {args.config}")

    # Run pipeline
    from pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(settings)
    state = orchestrator.run(
        data_path=str(data_path),
        target_column=args.target,
        run_id=args.run_id,
        resume=args.resume,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Run ID:       {state.run_id}")
    print(f"Status:       {'FAILED' if state.failed else 'SUCCESS'}")
    print(f"Problem Type: {state.problem_type}")
    print(f"Best Model:   {state.best_model_name or 'N/A'}")
    print(f"Best Metric:  {state.best_metric_name}={state.best_metric_value:.4f}" if state.best_metric_value else "Best Metric:  N/A")
    print(f"Retries:      {state.retry_count}")
    print(f"Artifacts:    {state.artifacts}")

    if state.report_paths:
        print(f"\nReports:")
        for name, path in state.report_paths.items():
            print(f"  {name}: {path}")

    if state.failed:
        print(f"\nFailure Reason: {state.failure_reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
