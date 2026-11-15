curl -X POST http://localhost:8000/api/generate/   -H "Content-Type: application/json"   -d '{
    "learningObjectives": [
      {"uuid": "lo-1", "text": "Explain how Arize phoenix works"}
    ],
    "questionTypes": ["MULTIPLE_CHOICE"],
    "difficultyLevels": ["Beginner","Intermediate"],
    "numberOfQuestions": 2,
    "locale": "en-US",
    "sourceText": "Arize Phoenix is an open-source LLM observability and evaluation platform that helps monitor, trace, and debug AI applications. It provides end-to-end tracing of LLM calls, agent workflows, retrieval operations, and evaluations, offering visibility into execution latency, token usage, model performance, and output quality. It supports OpenTelemetry-based instrumentation and integrates with frameworks such as LangChain, LangGraph, and OpenAI."
  }'