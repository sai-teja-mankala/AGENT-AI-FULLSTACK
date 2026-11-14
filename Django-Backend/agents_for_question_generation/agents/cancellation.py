"""Job cancellation hook. Django-Backend holds no job state, so nothing is ever cancelled here."""


def is_job_cancelled(task_id) -> bool:
    return False
