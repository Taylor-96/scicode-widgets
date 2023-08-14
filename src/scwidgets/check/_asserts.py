from ._check import Check

import functools
from typing import List, Union

import numpy as np


def assert_shape(
    output_parameters: Check.TupleFunOutParamsTs,
    output_references: Check.TupleFunOutParamsTs,
    parameter_indices: Union[List[int], str] = "auto",
) -> str:
    if parameter_indices == "auto":
        parameter_indices = []
        for i in range(len(output_parameters)):
            if hasattr(output_references[i], "shape"):
                parameter_indices.append(i)
    elif parameter_indices == "all":
        parameter_indices = range(len(output_parameters))
    elif isinstance(parameter_indices, str):
        raise ValueError(
            f'Got parameter_indices="{parameter_indices}" but only "all" '
            "is accepted as string"
        )

    for i in parameter_indices:
        if output_parameters[i].shape != output_references[i].shape:
            return (
                f"For parameter {i} expected shape {output_references[i].shape} "
                f"but got {output_parameters[i].shape}."
            )
    return ""


def assert_numpy_allclose(
    output_parameters: Check.TupleFunOutParamsTs,
    output_references: Check.TupleFunOutParamsTs,
    parameter_indices: Union[List[int], str] = "auto",
    rtol=1e-05,
    atol=1e-08,
    equal_nan=False,
) -> str:
    assert len(output_parameters) == len(
        output_references
    ), "output_parameters and output_references have to have the same length"
    if parameter_indices == "auto":
        # we determine if allclose can be applied by applying it on itself
        for i in range(len(output_references)):
            parameter_indices = []
            try:
                np.allclose(output_references[i], output_references[i])
                parameter_indices.append(i)
            except Exception:
                pass
    elif parameter_indices == "all":
        parameter_indices = range(len(output_parameters))
    elif isinstance(parameter_indices, str):
        raise ValueError(
            f'Got parameter_indices="{parameter_indices}" but only "all" '
            'and "auto" is accepted as string'
        )

    for i in parameter_indices:
        is_allclose = np.allclose(
            output_parameters[i],
            output_references[i],
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )

        if not (is_allclose):
            diff = np.abs(
                np.asarray(output_parameters[i]) - np.asarray(output_references[i])
            )
            abs_diff = np.sum(diff)
            rel_diff = np.sum(diff / np.abs(output_references[i]))
            return (
                f"Parameter {i} is not close to reference absolute difference is "
                f"{abs_diff}, relative difference is {rel_diff}."
            )
    return ""


def assert_type(
    output_parameters: Check.TupleFunOutParamsTs,
    output_references: Check.TupleFunOutParamsTs,
    parameter_indices: Union[List[int], str] = "all",
) -> str:
    if parameter_indices == "all":
        parameter_indices = range(len(output_parameters))
    elif isinstance(parameter_indices, str):
        raise ValueError(
            f'Got parameter_indices="{parameter_indices}" but only "all" is '
            "accepted as string"
        )

    for i in parameter_indices:
        if not (isinstance(output_parameters[i], type(output_references[i]))):
            return (
                f"Expected type {type(output_references)} "
                f"but got {type(output_parameters)}."
            )
    return ""


def assert_numpy_sub_dtype(
    output_parameters: Union[Check.TupleFunOutParamsTs, tuple[Check.FingerprintT]],
    dtype: np.dtype,
    parameter_indices: Union[List[int], str] = "all",
) -> str:
    if parameter_indices == "all":
        parameter_indices = range(len(output_parameters))
    if dtype is None:
        dtype = np.floating
    for i in parameter_indices:
        if not (isinstance(output_parameters[i], np.ndarray)):
            return (
                f"Output parameter {i} expected to be numpy array "
                f"but got {type(output_parameters[i])}."
            )
        if not (np.issubdtype(output_parameters[i].dtype, dtype)):
            return (
                f"Output parameter {i} expected to be sub dtype numpy.{dtype.__name__} "
                f"but got numpy.{output_parameters[i].dtype.type.__name__}."
            )
    return ""


assert_numpy_floating_sub_dtype = functools.partial(
    assert_numpy_sub_dtype, dtype=np.floating
)
