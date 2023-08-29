# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import sys

import scwidgets
from scwidgets.check import CheckRegistry

sys.path.insert(0, os.path.abspath(".."))
from test_check import mock_checkable_widget  # noqa: E402
from test_check import single_param_check  # noqa: E402

# -

scwidgets.get_css_style()


def test_check_registry(use_fingerprint, failing, buggy):
    check_registry = CheckRegistry()
    checkable_widget = mock_checkable_widget(check_registry)

    check = single_param_check(
        use_fingerprint=use_fingerprint, failing=failing, buggy=buggy
    )
    checkable_widget.compute_output_to_check = check.function_to_check

    checkable_widget.add_check(
        check.asserts,
        check.inputs_parameters,
        check.outputs_references,
        check.fingerprint,
    )
    return check_registry


# Test 1:
# -------
# Test if CheckRegistry shows correct output

# Test 1.1
test_check_registry(use_fingerprint=False, failing=False, buggy=False)

# Test 1.2
test_check_registry(use_fingerprint=True, failing=False, buggy=False)

# Test 1.3
test_check_registry(use_fingerprint=False, failing=True, buggy=False)

# Test 1.4
test_check_registry(use_fingerprint=True, failing=True, buggy=False)

# Test 1.5
test_check_registry(use_fingerprint=False, failing=False, buggy=True)