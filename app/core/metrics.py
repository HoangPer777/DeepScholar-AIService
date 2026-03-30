from prometheus_client import Counter, Histogram, Gauge

REQUEST_TOTAL = Counter("dr_request_total", "Total research requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("dr_request_latency_seconds", "Research request latency", ["endpoint"])
ACTIVE_REQUESTS = Gauge("dr_active_requests", "Current active requests", ["endpoint"])

LLM_CALL_TOTAL = Counter("dr_llm_call_total", "LLM call total", ["provider", "model", "status"])
TOOL_CALL_TOTAL = Counter("dr_tool_call_total", "Tool call total", ["tool", "status"])

RANKED_CONTEXT_SIZE = Histogram("dr_ranked_context_size", "How many context chunks survive rerank")
REVIEW_SCORE = Histogram("dr_review_score", "Reviewer confidence score")
