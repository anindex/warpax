import pytest
from warpax.exceptions import (
    AsymptoticFalloffError,
    ConstraintViolationError,
    JunctionDiscontinuityError,
    TOVInconsistencyError,
    TransportUndefinedError,
    WarpAXError,
)


def test_exception_inheritance():
    """All domain exceptions inherit from WarpAXError."""
    assert issubclass(AsymptoticFalloffError, WarpAXError)
    assert issubclass(ConstraintViolationError, WarpAXError)
    assert issubclass(JunctionDiscontinuityError, WarpAXError)
    assert issubclass(TOVInconsistencyError, WarpAXError)
    assert issubclass(TransportUndefinedError, WarpAXError)


@pytest.mark.parametrize(
    ("exc_cls", "kwargs", "expected_attrs", "expected_substrings"),
    [
        pytest.param(
            ConstraintViolationError,
            {"residual_type": "Hamiltonian", "max_residual": 1e-3, "location": "(0, 1, 0, 0)"},
            {"residual_type": "Hamiltonian", "max_residual": 1e-3, "location": "(0, 1, 0, 0)"},
            ["Hamiltonian", "1.000000e-03", "(0, 1, 0, 0)"],
            id="ConstraintViolationError",
        ),
        pytest.param(
            TOVInconsistencyError,
            {"max_residual": 2.5e-4, "r_location": 1.5},
            {"max_residual": 2.5e-4, "r_location": 1.5},
            ["2.500000e-04", "1.500000e+00"],
            id="TOVInconsistencyError",
        ),
        pytest.param(
            JunctionDiscontinuityError,
            {"surface_label": "r=10", "jump_magnitude": 0.01},
            {"surface_label": "r=10", "jump_magnitude": 0.01},
            ["r=10", "1.000000e-02"],
            id="JunctionDiscontinuityError",
        ),
        pytest.param(
            AsymptoticFalloffError,
            {"measured_order": 1.5, "required_order": 2},
            {"measured_order": 1.5, "required_order": 2},
            ["1.50", "2"],
            id="AsymptoticFalloffError",
        ),
        pytest.param(
            TransportUndefinedError,
            {"geodesic_id": 42, "termination_reason": "max_iter exceeded"},
            {"geodesic_id": 42, "termination_reason": "max_iter exceeded"},
            ["42", "max_iter exceeded"],
            id="TransportUndefinedError",
        ),
    ],
)
def test_exception_fields_and_message(
    exc_cls, kwargs, expected_attrs, expected_substrings
):
    """Each exception stores constructor parameters as attributes and formats them in the message."""
    err = exc_cls(**kwargs)
    for attr, expected in expected_attrs.items():
        assert getattr(err, attr) == expected
    msg = str(err)
    for substring in expected_substrings:
        assert substring in msg
