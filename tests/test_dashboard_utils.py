"""Dashboard utility smoke tests."""

from src.dashboard_utils import load_mock_data


def test_load_mock_data_generates_requested_records():
    data = load_mock_data(num_trades=8, num_debates=3, seed=1)

    assert len(data["trades"]) == 8
    assert len(data["debates"]) == 3
    assert not data["equity_curve"].empty
