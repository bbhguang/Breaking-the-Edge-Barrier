"""Budget calibration for degree collection and sampled wedge reports.

The shuffle expression is also evaluated outside its admissible local-budget
range when finding the preliminary wedge budget. That algebraic inverse is a
parameter-selection step, not an application of the amplification theorem.
Returned ``epsilon_bound`` values describe the calibration formula; they do
not independently establish a privacy guarantee for the graph protocol.
"""

import math
from numbers import Integral
import sys


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_domain(n, delta):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 2:
        raise ValueError("n must be an integer of at least two")
    delta = _positive(delta, "delta")
    if delta >= 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    return int(n), delta


def _log_expm1(value):
    """Return log(exp(value) - 1) without overflowing for large value."""
    if value > math.log(2.0):
        return value + math.log1p(-math.exp(-value))
    return math.log(math.expm1(value))


def _softplus(value):
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _logaddexp(left, right):
    larger, smaller = max(left, right), min(left, right)
    return larger + math.log1p(math.exp(smaller - larger))


def shuffle_threshold(n, delta):
    """Return log(n / (16 log(2/delta))), without clamping its sign."""
    n, delta = _validate_domain(n, delta)
    log_two_over_delta = math.log(2.0) - math.log(delta)
    return math.log(n) - math.log(16.0) - math.log(log_two_over_delta)


def _admissible_threshold(n, delta):
    threshold = shuffle_threshold(n, delta)
    if threshold <= 0.0:
        raise ValueError(
            "The shuffle threshold is not positive for this n and delta; "
            "the stated calibration cannot be applied"
        )
    return threshold


def shuffle_amplify(epsilon_local, n, delta):
    """Evaluate the shuffle expression using logarithmic arithmetic.

    Its theorem condition is ``epsilon_local <= shuffle_threshold(n, delta)``.
    This evaluator deliberately does not enforce that condition, because the
    preliminary wedge budget requires the unrestricted algebraic inverse.
    """
    n, delta = _validate_domain(n, delta)
    epsilon_local = _positive(epsilon_local, "epsilon_local")
    log_n = math.log(n)
    log_four_over_delta = math.log(4.0) - math.log(delta)
    first = math.log(8.0) + epsilon_local / 2.0 + (
        math.log(log_four_over_delta) - log_n
    ) / 2.0
    second = math.log(8.0) + epsilon_local - log_n
    if epsilon_local < 1e-5:
        # tanh(x/2) = (x/2) * (1 - x^2/12 + O(x^4)).
        log_ratio = math.log(epsilon_local) - math.log(2.0)
        log_ratio += math.log1p(-epsilon_local * epsilon_local / 12.0)
    else:
        log_ratio = math.log(math.tanh(epsilon_local / 2.0))
    return _softplus(log_ratio + _logaddexp(first, second))


def invert_shuffle(target, n, delta, cap_at_threshold=False):
    """Invert the increasing shuffle expression by bracketed bisection.

    With ``cap_at_threshold=True``, budgets above the theorem's admissible
    range are capped at that threshold. The degree phase uses this option;
    the preliminary wedge calibration must leave it disabled.
    """
    n, delta = _validate_domain(n, delta)
    target = _positive(target, "target")
    if cap_at_threshold:
        upper = _admissible_threshold(n, delta)
        if shuffle_amplify(upper, n, delta) <= target:
            return upper
    else:
        upper = target
        while shuffle_amplify(upper, n, delta) < target:
            if upper >= sys.float_info.max / 2.0:
                upper = sys.float_info.max
                if shuffle_amplify(upper, n, delta) < target:
                    raise ValueError("The inverse budget exceeds floating-point range")
                break
            upper *= 2.0
    lower = 0.0
    for _ in range(200):
        midpoint = lower + (upper - lower) / 2.0
        if midpoint == lower or midpoint == upper:
            break
        if shuffle_amplify(midpoint, n, delta) < target:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= 8.0 * sys.float_info.epsilon * upper:
            break
    result = lower + (upper - lower) / 2.0
    return result if result > 0.0 else upper


def calibrate_degree(target, n, delta):
    """Calibrate degree reports without the wedge phase's division by tau."""
    n, delta = _validate_domain(n, delta)
    target = _positive(target, "target")
    threshold = _admissible_threshold(n, delta)
    epsilon_local = invert_shuffle(target, n, delta, cap_at_threshold=True)
    return {
        "epsilon_local": epsilon_local,
        "epsilon_bound": shuffle_amplify(epsilon_local, n, delta),
        "threshold": threshold,
    }


def calibrate_wedge(target, n, delta, tau):
    """Calibrate sampling, allocate the effective budget, and invert sampling.

    Let a be the unrestricted inverse shuffle budget and theta the threshold.
    Choose p = min(1, expm1(theta)/expm1(a)), then q = log1p(p*expm1(a)).
    The per-user effective budget is q/tau. Keeping p fixed, invert the
    sampling expression to obtain the local noise budget. This differs from
    dividing a by tau whenever p is below one.
    """
    n, delta = _validate_domain(n, delta)
    target = _positive(target, "target")
    if isinstance(tau, bool) or not isinstance(tau, Integral) or not 1 <= tau < n:
        raise ValueError("tau must be an integer between one and n-1")
    tau = int(tau)
    threshold = _admissible_threshold(n, delta)
    preliminary = invert_shuffle(target, n, delta)
    if preliminary <= threshold:
        probability = 1.0
        log_probability = 0.0
        effective_joint = preliminary
    else:
        log_probability = _log_expm1(threshold) - _log_expm1(preliminary)
        probability = math.exp(log_probability)
        if probability == 0.0:
            raise ValueError("The sampling probability is below floating-point range")
        # With this choice of p, log1p(p*expm1(a)) equals theta exactly.
        effective_joint = threshold
    effective_user = effective_joint / tau
    if effective_user == 0.0:
        raise ValueError("The per-user effective budget is below floating-point range")
    noise_epsilon = (
        effective_user if probability == 1.0
        else _softplus(_log_expm1(effective_user) - log_probability)
    )
    noise_scale = 2.0 * tau / noise_epsilon
    if not math.isfinite(noise_scale):
        raise ValueError("The required Laplace scale exceeds floating-point range")
    return {
        "epsilon_joint_base": preliminary,
        "p": probability,
        "epsilon_joint_effective": effective_joint,
        "epsilon_effective_per_user": effective_user,
        "epsilon_noise": noise_epsilon,
        "noise_scale": noise_scale,
        "threshold": threshold,
        "epsilon_bound": shuffle_amplify(effective_joint, n, delta),
    }
