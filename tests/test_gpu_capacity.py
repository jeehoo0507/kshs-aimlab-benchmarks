"""The planner must scale past four jobs and preserve free VRAM."""

from scripts.gpu_capacity import job_budget_gib, memory_job_limit


def test_capacity_uses_measured_free_memory_without_fixed_job_ceiling():
    assert job_budget_gib(2.0) == 3.0
    assert memory_job_limit(20.0, 2.0) == 5
    assert memory_job_limit(2.5, 2.0) == 0
