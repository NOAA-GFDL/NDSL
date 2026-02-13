import time

import pytest

from ndsl.performance import NullTimer, Timer


@pytest.fixture
def timer():
    return Timer()


@pytest.fixture
def null_timer():
    return NullTimer()


@pytest.fixture
def disabled_timer():
    return Timer(enabled=False)


def test_start_stop(timer: Timer) -> None:
    timer.start("label")
    timer.stop("label")

    assert "label" in timer.times
    assert len(timer.times) == 1
    assert timer.hits["label"] == 1
    assert len(timer.hits) == 1


def test_null_timer_cannot_be_enabled(null_timer: NullTimer) -> None:
    with pytest.raises(NotImplementedError, match="`NullTimer` cannot be enabled."):
        null_timer.enable()


def test_null_timer_is_disabled(null_timer: NullTimer) -> None:
    assert not null_timer.enabled


def test_clock(timer: Timer) -> None:
    with timer.clock("label"):
        # small arbitrary computation task to time
        time.sleep(0.1)

    assert "label" in timer.times
    assert len(timer.times) == 1
    assert abs(timer.times["label"] - 0.1) < 1e-2
    assert timer.hits["label"] == 1
    assert len(timer.hits) == 1


def test_start_twice(timer: Timer) -> None:
    """Cannot call start() twice consecutively with no stop() in between."""
    timer.start("label")
    with pytest.raises(ValueError, match="Clock already started for .*"):
        timer.start("label")


def test_clock_in_clock(timer: Timer) -> None:
    """Should not be able to create a given clock inside itself."""
    with timer.clock("label"):
        with pytest.raises(ValueError, match="Clock already started for .*"):
            with timer.clock("label"):
                pass


def test_consecutive_start_stops(timer: Timer) -> None:
    """Total time increases with consecutive clock start/stop calls."""
    timer.start("label")
    time.sleep(0.01)
    timer.stop("label")
    previous_time = timer.times["label"]
    for _i in range(5):
        timer.start("label")
        time.sleep(0.01)
        timer.stop("label")
        assert timer.times["label"] >= previous_time + 0.01
        previous_time = timer.times["label"]
    assert timer.hits["label"] == 6


def test_consecutive_clocks(timer: Timer) -> None:
    """Total time increases with consecutive clock blocks."""
    with timer.clock("label"):
        time.sleep(0.01)
    previous_time = timer.times["label"]
    for _i in range(5):
        with timer.clock("label"):
            time.sleep(0.01)
        assert timer.times["label"] >= previous_time + 0.01
        previous_time = timer.times["label"]
    assert timer.hits["label"] == 6


@pytest.mark.parametrize(
    "operations, result",
    [
        ([], True),
        (["enable"], True),
        (["disable"], False),
        (["disable", "enable"], True),
        (["disable", "disable"], False),
    ],
)
def test_enable_disable(timer: Timer, operations: list[str], result: bool) -> None:
    for operation in operations:
        getattr(timer, operation)()
    assert timer.enabled == result


def test_disabled_timer_does_not_add_key(disabled_timer: Timer) -> None:
    with disabled_timer.clock("label1"):
        time.sleep(0.01)
    assert len(disabled_timer.times) == 0
    with disabled_timer.clock("label2"):
        time.sleep(0.01)
    assert len(disabled_timer.times) == 0
    assert len(disabled_timer.hits) == 0


def test_disabled_timer_does_not_add_time(timer: Timer) -> None:
    with timer.clock("label"):
        time.sleep(0.01)
    initial_time = timer.times["label"]
    timer.disable()
    with timer.clock("label"):
        time.sleep(0.01)
    assert timer.times["label"] == initial_time
    assert timer.hits["label"] == 1


@pytest.fixture(params=["clean", "one_label", "two_labels"])
def used_timer(request, timer) -> Timer:
    if request.param == "clean":
        return timer

    if request.param == "one_label":
        with timer.clock("label1"):
            time.sleep(0.01)
        return timer

    if request.param == "two_labels":
        with timer.clock("label1"):
            time.sleep(0.01)
        with timer.clock("label2"):
            time.sleep(0.01)
        return timer


def test_timer_reset(used_timer: Timer) -> None:
    used_timer.reset()
    assert len(used_timer.times) == 0
    assert len(used_timer.hits) == 0
