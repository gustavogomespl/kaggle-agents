"""Utility functions for ensemble operations."""

import numpy as np
import pandas as pd


def normalize_class_order(order: list) -> list[str]:
    """Normalize class order for comparison.

    Handles whitespace and encoding differences that cause false mismatches.
    Example: 'NGT - Incomplete' vs 'NGT - Incompletely Imaged' would still differ,
    but 'ETT - Abnormal ' (trailing space) would match 'ETT - Abnormal'.
    """
    if not order:
        return []
    return [str(c).strip() for c in order]


def class_orders_match(order1: list, order2: list) -> bool:
    """Compare two class orders with normalization.

    Returns True if orders are equivalent after normalization.
    """
    norm1 = normalize_class_order(order1)
    norm2 = normalize_class_order(order2)
    return norm1 == norm2


def encode_labels(
    y: np.ndarray | pd.Series, class_order: list[str] | None
) -> tuple[np.ndarray, list[str]]:
    """Encode labels to integer indices with optional class order.

    Args:
        y: Labels to encode
        class_order: Optional ordered list of class names

    Returns:
        Tuple of (encoded_labels, class_order)
    """
    y_array = np.asarray(y)

    if class_order:
        cat = pd.Categorical(y_array, categories=class_order)
        if (cat.codes < 0).any():
            print("   Warning: Label encoding mismatch with class_order, using sorted uniques")
        else:
            return cat.codes.astype(int), class_order

    classes, y_encoded = np.unique(y_array, return_inverse=True)
    return y_encoded.astype(int), classes.tolist()
