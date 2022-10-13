"""Provides a number of commonly used validation routines across the analysis classes."""

from __future__ import annotations

import attrs
import numpy as np


def validate_UQ_input(cls, attribute: attrs.Attribute, value: float | tuple) -> None:
    """Validates values that should be a float when :py:attr:`UQ` is False, or a 2-tuple of floats
    when :py:attr:`UQ` is False.

    Args:
        attribute (attrs.Attribute): The attrs Attribute information for the class attribute being
            validated.
        value (float | tuple): The user input to the class attribute.

    Raises:
        ValueError: Raised if any of the following occur:
             - If UQ is True, and value is not a tuple
             - If UQ is True, and value is not a length-2 tuple
             - If UQ is True, and each value is not a float
             - If UQ is False, and the value is not a float.
    """
    if cls.UQ:
        if not isinstance(value, tuple):
            raise ValueError(f"When UQ is True, {attribute.name} must be a tuple of length 2.")
        if len(value) != 2:
            raise ValueError(f"When UQ is True, {attribute.name} must be a tuple of length 2.")
        if not all(isinstance(x, (float, int)) for x in value):
            raise ValueError(f"All values of {attribute.name} must be of type 'float'.")
    else:
        if not isinstance(value, float):
            raise ValueError(
                f"When UQ is False, the value provided to {attribute.name} ({value}), must be a float"
            )


def validate_open_range_0_1(cls, attribute: attrs.Attribute, value: float | tuple) -> None:
    """Validates that the value, or tuple of values is in the half-closed range of (0, 1].

    Args:
        attribute (attrs.Attribute): The attrs Attribute information for the class attribute being
            validated.
        value (float | tuple): The user input to the class attribute.

    Raises:
        ValueError: Raised if a single input is passed and outside the range of (0, 1].
        ValueError: Raised if any of the inputs in the input tuple are outside the range of (0, 1].
    """
    if isinstance(value, float):
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"The value provided to '{attribute.name}' ({value}) must be in the range (0, 1]."
            )
    else:
        if not all(0.0 < x <= 1.0 for x in value):
            raise ValueError(
                f"The values provided to '{attribute.name}' ({value}) must be in the range (0, 1]."
            )
