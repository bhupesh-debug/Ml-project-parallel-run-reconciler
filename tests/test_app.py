"""Smoke test for the Streamlit review app (skips if outputs are absent)."""
import os

import pytest

pytest.importorskip("streamlit")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "reports", "review_list.csv")),
    reason="pipeline outputs not present; run python -m src.run_pipeline first",
)
def test_app_runs_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1  # the ranked review list renders
