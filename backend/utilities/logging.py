"""Universal logging utilities for any domain."""

import logging
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime


class LoggingUtils:
    """Universal logging utilities that work across all domains."""

    @staticmethod
    def setup_logger(
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        format_json: bool = False
    ) -> logging.Logger:
        """Set up a logger with universal configuration."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create formatter
        if format_json:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def log_extraction_metrics(
        logger: logging.Logger,
        domain: str,
        num_texts: int,
        num_entities: int,
        num_relations: int,
        duration: float
    ) -> None:
        """Log extraction metrics."""
        logger.info(
            "Extraction completed",
            extra={
                'domain': domain,
                'num_texts': num_texts,
                'num_entities': num_entities,
                'num_relations': num_relations,
                'duration_seconds': duration,
                'entities_per_text': num_entities / max(num_texts, 1),
                'relations_per_text': num_relations / max(num_texts, 1)
            }
        )

    @staticmethod
    def log_query_metrics(
        logger: logging.Logger,
        query_id: str,
        domain: str,
        query_type: str,
        num_results: int,
        duration: float,
        confidence: float
    ) -> None:
        """Log query metrics."""
        logger.info(
            "Query processed",
            extra={
                'query_id': query_id,
                'domain': domain,
                'query_type': query_type,
                'num_results': num_results,
                'duration_seconds': duration,
                'confidence': confidence
            }
        )

    @staticmethod
    def log_error_with_context(
        logger: logging.Logger,
        error: Exception,
        context: dict
    ) -> None:
        """Log error with additional context."""
        logger.error(
            f"Error occurred: {str(error)}",
            extra={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context
            },
            exc_info=True
        )


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                              'pathname', 'filename', 'module', 'lineno',
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'stack_info', 'exc_info', 'exc_text', 'message']:
                    log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry)