"""Output content guardrails (warn-only). Placeholder: no-op until a real check is configured."""


class _Result:
    def __init__(self, passed=True, flagged_content=None):
        self.passed = passed
        self.flagged_content = flagged_content


class ContentGuardrails:
    def check_all_parallel(self, content, lo_description, source_text):
        return {}


def run_output_guardrails(questions, lo_description, source_text, regenerate_fn=None, logger_instance=None):
    """Returns (questions, warning). Warning is None when nothing is flagged."""
    return questions, None
