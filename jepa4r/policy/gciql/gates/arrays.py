"""Full-array comparisons; no model, optimizer, or simulator dependencies."""

from collections.abc import Mapping
import hashlib
import math
from numbers import Integral, Real

import numpy as np


def array_identity(array):
    """The exact identity of an ordinary array: dtype, shape and the sha256 of its bytes."""
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise ValueError("object arrays have no stable byte identity")
    prefix = (str(value.dtype) + str(value.shape)).encode("utf-8")
    return hashlib.sha256(prefix + np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _tolerances(atol, rtol):
    if not all(isinstance(x, Real) and math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")


def compare_array(actual, reference, *, atol=1e-6, rtol=1e-4, exact=False):
    """Compare every element; the shapes and dtypes must match exactly."""
    _tolerances(atol, rtol)
    actual, reference = np.asarray(actual), np.asarray(reference)
    result = dict(
        passed=False,
        actual_sha256=array_identity(actual),
        reference_sha256=array_identity(reference),
        shape=list(reference.shape),
        dtype=str(reference.dtype),
        atol=atol,
        rtol=rtol,
        out_of_tolerance=None,
        max_abs_error=None,
        count=reference.size,
    )
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        result["reason"] = "shape_or_dtype_mismatch"
        return result
    if actual.dtype.kind not in "biuf":
        raise ValueError("numeric comparison requires boolean, integer, or real arrays")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        result["reason"] = "nonfinite_values"
        return result
    if actual.dtype.kind in "biu":
        # Integer and boolean quantities are exact; avoid uint64 subtraction/float64 loss.
        difference = np.abs(actual.astype(object) - reference.astype(object))
        outside = actual != reference
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
            outside = difference > (0.0 if exact else atol + rtol * np.abs(reference.astype(np.float64)))
    result.update(
        out_of_tolerance=int(np.count_nonzero(outside)),
        max_abs_error=float(np.max(difference)) if reference.size else 0.0,
        passed=not bool(np.any(outside)),
    )
    result["reason"] = "equal" if result["passed"] else "values_out_of_tolerance"
    return result


def compare_array_maps(actual, reference, **options):
    """Missing/extra gradient or parameter names are failures, never ignored."""
    if not isinstance(actual, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("array maps must be mappings")
    missing, extra = set(reference) - set(actual), set(actual) - set(reference)
    checks = {name: compare_array(actual[name], reference[name], **options) for name in reference if name in actual}
    return dict(
        passed=bool(reference) and not missing and not extra and all(x["passed"] for x in checks.values()),
        missing=[str(x) for x in sorted(missing, key=repr)],
        extra=[str(x) for x in sorted(extra, key=repr)],
        arrays=checks,
    )


def compare_parameter_updates(
    actual_before, actual_after, reference_before, reference_after, *, relative_l2=1e-3, cosine=0.99999
):
    """Compare complete increments from the same initial parameter state."""
    _tolerances(relative_l2, 0.0)
    if not isinstance(cosine, Real) or not math.isfinite(cosine) or not -1 <= cosine <= 1:
        raise ValueError("cosine threshold must be finite and in [-1,1]")
    maps = (actual_before, actual_after, reference_before, reference_after)
    if not all(isinstance(m, Mapping) for m in maps):
        raise ValueError("parameter states must be mappings")
    keys = set(reference_before)
    result = dict(
        passed=False,
        relative_l2=None,
        cosine=None,
        relative_l2_limit=relative_l2,
        cosine_min=cosine,
        parameters=len(keys),
        elements=0,
    )
    if not keys or any(set(m) != keys for m in maps):
        result["reason"] = "empty_or_mismatched_parameter_names"
        return result
    initial = compare_array_maps(actual_before, reference_before, exact=True)
    result["initial_state"] = initial
    if not initial["passed"]:
        result["reason"] = "different_initial_state"
        return result
    actual_delta, reference_delta = [], []
    for name in sorted(keys):
        values = [np.asarray(m[name]) for m in maps]
        if any(
            v.shape != values[0].shape or v.dtype != values[0].dtype or v.dtype.kind != "f" or not np.isfinite(v).all()
            for v in values
        ):
            result["reason"] = "invalid_parameter_shape_dtype_or_values"
            return result
        actual_delta.append((values[1].astype(np.float64) - values[0].astype(np.float64)).ravel())
        reference_delta.append((values[3].astype(np.float64) - values[2].astype(np.float64)).ravel())
    actual_delta, reference_delta = np.concatenate(actual_delta), np.concatenate(reference_delta)
    actual_norm, reference_norm = np.linalg.norm(actual_delta), np.linalg.norm(reference_delta)
    result.update(
        elements=int(reference_delta.size),
        actual_delta_norm=float(actual_norm),
        reference_delta_norm=float(reference_norm),
    )
    if not (math.isfinite(actual_norm) and math.isfinite(reference_norm) and actual_norm > 0 and reference_norm > 0):
        result["reason"] = "no_finite_effective_update"
        return result
    relative = float(np.linalg.norm(actual_delta - reference_delta) / reference_norm)
    direction = float(np.clip(np.dot(actual_delta / actual_norm, reference_delta / reference_norm), -1.0, 1.0))
    result.update(relative_l2=relative, cosine=direction, passed=relative <= relative_l2 and direction >= cosine)
    result["reason"] = "update_matches" if result["passed"] else "update_out_of_tolerance"
    return result


def compare_nested(actual, reference, *, atol=1e-6, rtol=1e-4, exact=True):
    """Compare optimizer/RNG trees without dropping keys, arrays, or sequence order."""
    _tolerances(atol, rtol)
    failures = []

    def visit(left, right, path):
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                failures.append(dict(path=path, reason="array_type_mismatch"))
                return
            check = compare_array(left, right, atol=atol, rtol=rtol, exact=exact)
            if not check["passed"]:
                failures.append(dict(path=path, comparison=check))
        elif isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                failures.append(dict(path=path, reason="mapping_type_mismatch"))
                return
            lk, rk = {(type(k), k) for k in left}, {(type(k), k) for k in right}
            if lk != rk:
                failures.append(dict(path=path, reason="mapping_key_mismatch"))
                return
            for key in right:
                visit(left[key], right[key], path + "/" + repr(key))
        elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                failures.append(dict(path=path, reason="sequence_mismatch"))
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, path + "/" + str(index))
        elif isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
            if not isinstance(left, (bool, np.bool_)) or not isinstance(right, (bool, np.bool_)) or left != right:
                failures.append(dict(path=path, reason="boolean_mismatch"))
        elif isinstance(left, Integral) and isinstance(right, Integral):
            if left != right:
                failures.append(dict(path=path, reason="integer_mismatch"))
        elif isinstance(left, Real) and isinstance(right, Real):
            check = compare_array(np.asarray(left), np.asarray(right), atol=atol, rtol=rtol, exact=exact)
            if not check["passed"]:
                failures.append(dict(path=path, comparison=check))
        elif type(left) is not type(right) or left != right:
            failures.append(dict(path=path, reason="value_or_type_mismatch"))

    visit(actual, reference, "$")
    return dict(passed=not failures, failures=failures)
