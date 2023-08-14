import re

import numpy as np
import pytest

from scwidgets.check import (
    Check,
    assert_numpy_allclose,
    assert_numpy_floating_sub_dtype,
    assert_shape,
    assert_type,
)


def test_assert_shape():
    output_parameters = (np.array([1, 2, 3]),)
    output_references = (np.array([1, 2, 3]),)
    result = assert_shape(output_parameters, output_references)
    assert result == ""


def test_assert_invalid_parameter_indices():
    with pytest.raises(
        ValueError,
        match=r"Got parameter_indices=\"invalid\" but only"
        ' "all" is accepted as string',
    ):
        assert_shape([1, 2, 3], [1, 2, 3], parameter_indices="invalid")


def test_assert_numpy_allclose():
    output_parameters = (np.array([1.0, 2.0]),)
    output_references = (np.array([1.1, 2.2]),)
    result = assert_numpy_allclose(output_parameters, output_references)
    assert "Parameter 0 is not close to reference" in result


def test_assert_type():
    output_parameters = (42,)
    output_references = (42,)
    result = assert_type(output_parameters, output_references)
    assert result == ""


def test_assert_numpy_floating_sub_dtype():
    output_parameters = (np.array([1.0, 2.0]),)
    result = assert_numpy_floating_sub_dtype(output_parameters)
    assert result == ""


def test_assert_invalid_output_parameter_dtype():
    output_parameters = (np.array([1, 2]),)
    message = assert_numpy_floating_sub_dtype(output_parameters)
    assert message == (
        "Output parameter 0 expected to be sub dtype numpy.floating "
        "but got numpy.int64."
    )


def single_input_check():
    def function_to_check(parameter):
        return parameter * 2

    return Check(
        function_to_check=function_to_check,
        asserts=[
            assert_type,
            assert_shape,
            assert_numpy_floating_sub_dtype,
            assert_numpy_allclose,
        ],
        inputs_parameters=[
            {"parameter": np.array([1.0])},
            {"parameter": np.array([2.0])},
        ],
        outputs_references=[(np.array([2.0]),), (np.array([4.0]),)],
        fingerprint=None,
    )


def multi_input_check():
    def function_to_check(parameter1, parameter2, parameter3=None):
        return parameter1 + parameter2, parameter3 * parameter2

    return Check(
        function_to_check=function_to_check,
        asserts=[
            assert_type,
            assert_shape,
            assert_numpy_floating_sub_dtype,
            assert_numpy_allclose,
        ],
        inputs_parameters=[
            {
                "parameter1": np.array([1.0]),
                "parameter2": np.array([2.0]),
                "parameter3": np.array([3.0]),
            }
        ],
        outputs_references=[
            (
                np.array([3.0]),
                np.array([6.0]),
            )
        ],
        fingerprint=None,
    )


def single_input_fingerprint_check():
    def function_to_check(parameter):
        return parameter * 2

    def fingerprint(parameter):
        return np.sum(parameter)

    return Check(
        function_to_check=function_to_check,
        asserts=[assert_numpy_floating_sub_dtype, assert_numpy_allclose],
        inputs_parameters=[
            {"parameter": np.array([1.0, 2.0])},
        ],
        outputs_references=[(6.0,)],
        fingerprint=fingerprint,
    )


@pytest.mark.parametrize(
    "check",
    [single_input_check(), multi_input_check(), single_input_fingerprint_check()],
)
def test_check_check_code(check):
    result = check.check_code()
    assert result == ""


def test_check_invalid_asserts_arguments_count():
    def function_to_check(parameter):
        return parameter * 2

    with pytest.raises(
        ValueError,
        match=r"Only assert function with 1 or 2 positional arguments are allowed",
    ):
        Check(
            function_to_check=function_to_check,
            asserts=[lambda output, ref, invalid: None],  # Three arguments
            inputs_parameters=[{"parameter": np.array([1])}],
            outputs_references=[(np.array([2]),)],
            fingerprint=None,
        ).check_code()


def test_check_mismatching_parameters_references_length():
    def function_to_check(parameter):
        return parameter * 2

    error_message = re.escape(
        "Number of output parameters and reference output parameters are mismatching: "
        "len output parameters != len outputs references [1 != 2]."
    )
    with pytest.raises(AssertionError, match=error_message):
        Check(
            function_to_check=function_to_check,
            asserts=[assert_shape],
            inputs_parameters=[{"parameter": np.array([1])}],
            outputs_references=[(np.array([2]), "invalid")],
            fingerprint=None,
        ).check_code()
