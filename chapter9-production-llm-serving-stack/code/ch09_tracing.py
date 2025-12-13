"""
OpenTelemetry instrumentation example.
"""
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Instrument code
# Note: This is a snippet - integrate into your FastAPI app
def generate_with_tracing(request: dict):
    """Example function with tracing"""
    with tracer.start_as_current_span("generate") as span:
        span.set_attribute("model", request["model"])
        
        with tracer.start_as_current_span("tokenize"):
            tokens = tokenize(request["prompt"])
        
        with tracer.start_as_current_span("inference") as inf_span:
            result = model_inference(tokens)
            inf_span.set_attribute("tokens", result["tokens"])
        
        return result

# Placeholder functions
def tokenize(prompt):
    return {"tokens": [1, 2, 3]}

def model_inference(tokens):
    return {"tokens": 10, "text": "generated text"}

if __name__ == "__main__":
    request = {"model": "llama-2-7b", "prompt": "Hello"}
    result = generate_with_tracing(request)
    print(result)

