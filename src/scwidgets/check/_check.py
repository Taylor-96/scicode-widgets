import inspect
from typing import Callable, List, Optional, TypeVar, TypeVarTuple, Union, Unpack


class Check:
    """
    We predefine the types here to use them in the member functions
    """

    FunInParamsTs = TypeVarTuple("FunInParamsTs")
    TupleFunInParamsTs = tuple[Unpack[FunInParamsTs]]
    FunOutParamsTs = TypeVarTuple("FunOutParamsTs")
    TupleFunOutParamsTs = tuple[Unpack[FunOutParamsTs]]
    FingerprintT = TypeVar("FingerprintT")


class Check:
    """
    A check verifies the correctness of a function for a set of inputs parameters using
    a list of univariate and bivariate asserts with the option to obscure the reference
    outputs.

    :param function_to_check:
        The code_obj must have a `compute_output_to_check` function that accepts each
        input parameters in :params input parameters:
    :param inputs_parameters:
        A dict or a list of dictionaries each containing the argument name and its value
        as (key, value) pair that is used as input for the function
        `compute_output_to_check` of
        :param code_obj:
    :param outputs_references:
        A list or a list of lists each containing the expected output of the function
        `compute_output_to_check` of :param code_obj: for the inputsin the :param
        input_parameters:
    :param asserts:
        A list of assert functions. An assert function can the output parameters of
        :param function_to_check: to run assert. If output references has been set it
        can take additional output references to compare with. If a fingerprint is given
        then the fingerprints are compared while assert functions with a single argument
        are always applied on the output paramaters.
    :param fingerprint:
        A one-way function that takes as input the output parameters of function :param
        function_to_check: and obscures the :param output_references:.
    """

    FunInParamsTs = Check.FunInParamsTs
    TupleFunInParamsTs = Check.TupleFunInParamsTs
    FunOutParamsTs = Check.FunOutParamsTs
    TupleFunOutParamsTs = Check.TupleFunOutParamsTs
    FingerprintT = Check.FingerprintT

    def __init__(
        self,
        function_to_check: Callable[[TupleFunInParamsTs], Check.TupleFunOutParamsTs],
        asserts: List[
            Union[
                Callable[[Check.TupleFunOutParamsTs, Check.TupleFunOutParamsTs], str],
                Callable[[tuple[Check.FingerprintT], tuple[Check.FingerprintT]], str],
                Callable[[Check.TupleFunOutParamsTs], str],
            ]
        ],
        inputs_parameters: Union[List[dict], dict],  # TODO make dict
        outputs_references: Optional[
            Union[  # TODO Check.TupleFunOutParamsTs
                List[Check.TupleFunOutParamsTs], Check.TupleFunOutParamsTs
            ]
        ] = None,
        fingerprint: Optional[
            Callable[[Unpack[Check.TupleFunOutParamsTs]], Check.FingerprintT]
        ] = None,
    ):
        self._function_to_check = function_to_check
        self._asserts = []
        self._univariate_asserts = []
        self._bivariate_asserts = []

        for i, assert_f in enumerate(asserts):
            nb_positional_arguments = len(
                [
                    parameters
                    for parameters in inspect.signature(assert_f).parameters.values()
                    if parameters.default is inspect._empty
                ]
            )
            self._asserts.append(assert_f)
            if nb_positional_arguments == 1:
                self._univariate_asserts.append(assert_f)
            elif nb_positional_arguments == 2:
                self._bivariate_asserts.append(assert_f)
            else:
                raise ValueError(
                    f"Only assert function with 1 or 2 positional arguments are allowed"
                    f"but assert function {i} has {nb_positional_arguments} positional"
                    f"arguments"
                )

        # We sadly cannot verify if the number of input argumets match because they can
        # be hidden in **kwargs
        if isinstance(inputs_parameters, dict):
            inputs_parameters = [inputs_parameters]

        if outputs_references is not None:
            if isinstance(outputs_references, tuple):
                outputs_references = [outputs_references]
            assert len(inputs_parameters) == len(outputs_references), (
                "Number of inputs_parameters and outputs_references are mismatching: "
                "len inputs parameters != len outputs parameters "
                f"[{len(inputs_parameters)} != {len(outputs_references)}]."
            )

        self._inputs_parameters = inputs_parameters
        self._outputs_references = outputs_references
        self._fingerprint = fingerprint

    @property
    def function_to_check(self):
        return self._function_to_check

    @function_to_check.setter
    def function_to_check(self, function_to_check):
        self._function_to_check = function_to_check

    @property
    def fingerprint(self):
        return self._fingerprint

    @property
    def asserts(self):
        return self._asserts

    @property
    def univariate_asserts(self):
        return self._univariate_asserts

    @property
    def bivariate_asserts(self):
        return self._bivariate_asserts

    @property
    def inputs_parameters(self):
        return self._inputs_parameters

    @property
    def outputs_references(self):
        return self._outputs_references

    def compute_outputs(self):
        outputs = []
        for input_parameters in self._inputs_parameters:
            output = self._function_to_check(**input_parameters)
            if not (isinstance(output, tuple)):
                output = (output,)
            if self._fingerprint is not None:
                output = self._fingerprint(*output)
                if not (isinstance(output, tuple)):
                    output = (output,)
            outputs.append(output)
        return outputs

    def compute_and_set_references(self):
        self._outputs_references = self.compute_outputs()

    def check_code(self) -> str:
        if len(self._bivariate_asserts) > 0:
            if self._outputs_references is None:
                raise ValueError(
                    "outputs_references are None but asserts exist that require "
                    "outputs_references (second positional argument)"
                )
            assert len(self._inputs_parameters) == len(self._outputs_references), (
                "Number of inputs and reference outputs  "
                "are mismatching: len inputs parameters != len outputs parameters "
                f"[{len(self._inputs_parameters)} != {len(self._outputs_references)}]."
            )

        outputs_message = ""
        for i in range(len(self._outputs_references)):
            output_message = ""
            for i, input_parameters in enumerate(self._inputs_parameters):
                output = self._function_to_check(**input_parameters)
                if not (isinstance(output, tuple)):
                    output = (output,)

                for assert_f in self._univariate_asserts:
                    assert_message = assert_f(output)
                    if assert_message:
                        output_message += (
                            f"CheckError: for input {i}:\n"
                            f"  {input_parameters}\n\n"
                            f"  {assert_f.__name__} failed: {assert_message}\n"
                        )

                if self._fingerprint is not None:
                    output = self._fingerprint(*output)
                    if not (isinstance(output, tuple)):
                        output = (output,)

                for assert_f in self._bivariate_asserts:
                    assert len(output) == len(self._outputs_references[i]), (
                        "Number of output parameters and reference output parameters "
                        "are mismatching: "
                        "len output parameters != len outputs references "
                        f"[{len(output)} != {len(self._outputs_references[i])}]."
                    )

                    assert_message = assert_f(output, self._outputs_references[i])
                    if assert_message:
                        output_message += (
                            f"CheckError: for input {i}:\n"
                            f"  {input_parameters}\n"
                            f"  {assert_f.__name__} failed: {assert_message}\n"
                        )
            outputs_message += output_message
        return outputs_message
