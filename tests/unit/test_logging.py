"""Tests for structured logging configuration."""

import logging

import structlog

from food_cpg_intelligence.logging import configure_logging, get_logger


def test_configure_logging_dev_mode() -> None:
    configure_logging(json_output=False, level="DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1


def test_configure_logging_json_mode() -> None:
    configure_logging(json_output=True, level="WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING


def test_get_logger_returns_usable_logger() -> None:
    configure_logging()
    logger = get_logger("test")
    # The proxy resolves to BoundLogger on first use. Bind a key to
    # prove it behaves as a real structured logger.
    bound = logger.bind(request_id="abc")
    assert isinstance(bound, structlog.stdlib.BoundLogger)
