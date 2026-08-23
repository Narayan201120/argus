"""Opt-in OpenTelemetry tracing (zero overhead when disabled).

TRACING_ENABLED=false (default): every helper here is a no-op and no
opentelemetry machinery is imported beyond this module's own lazy paths.
When enabled, spans are exported per the console exporter (local viewing)
or an OTLP endpoint (Jaeger/Grafana Tempo etc.; requires installing the
``opentelemetry-exporter-otlp`` extra manually).
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_configured = False


def configure_tracing(exporter_factory=None) -> None:
    """Install the SDK tracer provider once, per settings.

    ``exporter_factory`` lets tests inject a custom SpanExporter; when
    omitted the exporter is chosen from TRACING_EXPORTER.
    """
    global _configured
    if not settings.tracing_enabled:
        return
    if _configured:
        return

    resource = Resource.create({
        "service.name": settings.app_name.lower(),
        "service.version": settings.app_version,
    })
    provider = TracerProvider(resource=resource)

    if exporter_factory is not None:
        exporter = exporter_factory()
    else:
        exporter_name = settings.tracing_exporter.lower()
        if exporter_name == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(
                    endpoint=settings.tracing_otlp_endpoint, insecure=True
                )
            except ImportError:
                logger.error({
                    "message": (
                        "OTLP exporter requested but package missing; "
                        "install opentelemetry-exporter-otlp. Falling back to console."
                    ),
                })
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()

    processor = BatchSpanProcessor(exporter)
    if settings.tracing_exporter.lower() == "console":
        # Console viewers want spans immediately; batch still flushes fast.
        processor = BatchSpanProcessor(exporter, schedule_delay_millis=200)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _configured = True
    logger.info({
        "message": "Tracing enabled",
        "exporter": exporter_name if exporter_factory is None else "custom",
        "endpoint": settings.tracing_otlp_endpoint,
    })


def get_tracer():
    """The ARGUS tracer (valid only after configure_tracing())."""
    return trace.get_tracer("argus")


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Record an OpenTelemetry span, or yield None when tracing is off."""
    if not settings.tracing_enabled or not _configured:
        yield None
        return

    start = time.perf_counter()
    with get_tracer().start_as_current_span(name) as current:
        if attributes:
            for key, value in attributes.items():
                current.set_attribute(key, value)
        try:
            yield current
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            current.set_attribute("argus.duration_ms", round(elapsed_ms, 2))
