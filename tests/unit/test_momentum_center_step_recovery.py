import pytest

from core.full_dipole_history import SourcePositionError
from core.momentum_center import VelocityDomainError
from core import momentum_center_pair as pair


def test_position_budget_can_split_without_changing_total_interval(monkeypatch):
    def advance(payload, width, count):
        if width > 0.5:
            raise SourcePositionError("position accuracy")
        return {"time": payload["time"] + width}, [width]

    monkeypatch.setattr(pair, "advance_pair", advance)
    source = {"time": 0.0}
    result, records = pair.advance_pair_refined(source, 1.0, 1)
    assert result == {"time": 1.0}
    assert records == [0.5, 0.5]
    assert source == {"time": 0.0}


@pytest.mark.parametrize(
    "error", [VelocityDomainError("velocity domain"), ValueError("causal history gap")]
)
def test_physical_or_causal_failures_are_not_retried(monkeypatch, error):
    calls = []

    def advance(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(pair, "advance_pair", advance)
    with pytest.raises(type(error)):
        pair.advance_pair_refined({}, 1.0, 3)
    assert len(calls) == 1


def test_failed_second_half_does_not_publish_partial_interval(monkeypatch):
    def advance(payload, width, count):
        if width > 0.5 or payload["time"] > 0:
            raise SourcePositionError("position accuracy")
        return {"time": 0.5}, [0.5]

    monkeypatch.setattr(pair, "advance_pair", advance)
    source = {"time": 0.0}
    with pytest.raises(SourcePositionError):
        pair.advance_pair_refined(source, 1.0, 1)
    assert source == {"time": 0.0}
