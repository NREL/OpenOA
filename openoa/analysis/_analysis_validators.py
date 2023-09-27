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
        # For some defaults values of analysis classes, a tuple is provided, so ensure that those
        # are correctly converted to mean value with a 2-decimal precision as is intended through
        # the original API, otherwise raise an error that a float should be provided.
        if isinstance(value, tuple):
            if len(value) == 2:
                object.__setattr__(cls, attribute.name, round(np.mean(value), 2))
        elif not isinstance(value, (int, float)):
            raise ValueError(
                f"When UQ is False, the value provided to {attribute.name} ({value}), must be a float"
            )


def validate_half_closed_0_1_right(cls, attribute: attrs.Attribute, value: float | tuple) -> None:
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


def validate_half_closed_0_1_left(cls, attribute: attrs.Attribute, value: float | tuple) -> None:
    """Validates that the value, or tuple of values is in the half-closed range of [0, 1).

    Args:
        attribute (attrs.Attribute): The attrs Attribute information for the class attribute being
            validated.
        value (float | tuple): The user input to the class attribute.

    Raises:
        ValueError: Raised if a single input is passed and outside the range of [0, 1).
        ValueError: Raised if any of the inputs in the input tuple are outside the range of [0, 1).
    """
    if isinstance(value, float):
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"The value provided to '{attribute.name}' ({value}) must be in the range (0, 1]."
            )
    else:
        if not all(0.0 <= x < 1.0 for x in value):
            raise ValueError(
                f"The values provided to '{attribute.name}' ({value}) must be in the range (0, 1]."
            )


def validate_reanalysis_selections(
    cls, attribute: attrs.Attribute, value: list[str] | None
) -> None:
    """Validates the inputs to ``reanalysis_products``, and if ``None`` is proviced, the associated
    ``PlantData`` object's available reanalyis products are provided.

    Args:
        attribute (attrs.Attribute): The attribute data for :py:attr:`value`.
        value (list[str] | None): The user-provided values to the class attribute.

    Raises:
        ValueError: Raised if "prodcut" is used in :py:attr:`reanalysis_products`.
        ValueError: Raised if a reanalysis product key that doesn't exist in the base ``PlantData``
            object is provided.
    """
    valid = [*cls.plant.reanalysis]
    if None in value or value is None:
        object.__setattr__(cls, "reanalysis_products", valid)
        return
    if "product" in value:
        raise ValueError(
            "Neither `plant.reanalysis` nor `reanalysis_products` can have 'product',"
            " as an input. 'product' is the empty default value and is reserved."
        )
    invalid = list(set(value).difference(valid))
    if invalid:
        raise ValueError(
            f"The following input to `reanalysis_products`: {invalid} are not contained"
            f" in `plant.reanalysis`: {valid}"
        )
