import pytest

from factr_teleop.gripper_feedback import torsional_spring_torque_nm


def spring(
    position_rad,
    *,
    opened=1.0,
    closed=2.0,
    velocity_rad_s=0.0,
    stiffness=0.08,
    damping=0.0,
    max_torque=0.05,
):
    return torsional_spring_torque_nm(
        position_rad,
        velocity_rad_s,
        opened,
        closed,
        stiffness,
        damping,
        max_torque,
    )


def test_spring_is_zero_at_and_beyond_open():
    assert spring(1.0) == 0.0
    assert spring(0.5) == 0.0


@pytest.mark.parametrize(
    ("opened", "closed", "position", "expected"),
    [
        (1.0, 2.0, 1.5, -0.04),
        (2.0, 1.0, 1.5, 0.04),
    ],
)
def test_spring_torque_always_points_toward_open(
    opened, closed, position, expected
):
    assert spring(position, opened=opened, closed=closed) == pytest.approx(expected)


def test_spring_displacement_stops_growing_past_calibrated_closed_endpoint():
    assert spring(3.0, max_torque=1.0) == pytest.approx(-0.08)


def test_spring_adds_angular_damping_only_while_engaged():
    assert spring(1.5, velocity_rad_s=1.0, damping=0.01) == pytest.approx(-0.05)
    assert spring(1.0, velocity_rad_s=1.0, damping=0.01) == 0.0


def test_spring_final_torque_is_limited():
    assert spring(2.0, stiffness=10.0, max_torque=0.03) == pytest.approx(-0.03)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"opened": 1.0, "closed": 1.0},
        {"stiffness": -0.1},
        {"damping": -0.1},
        {"max_torque": 0.0},
        {"position_rad": float("nan")},
    ],
)
def test_invalid_spring_parameters_are_rejected(kwargs):
    position_rad = kwargs.pop("position_rad", 1.5)
    with pytest.raises(ValueError):
        spring(position_rad, **kwargs)
