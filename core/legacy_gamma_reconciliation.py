"""Explicit, deprecated interpretation of saved gamma-reconciliation studies."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum, auto
from functools import wraps


class GammaReconciliationMethod(Enum):
    """Deprecated choices for interpreting saved gamma-reconciliation studies.

    Ordinary runs derive gamma and beta from on-shell spatial momentum.
    These choices only preserve the historical iteration-seed interpretation;
    they are not accuracy or stability controls.
    """

    DISABLED = auto()
    ADAPTIVE_WEIGHTED = auto()
    USE_VELOCITY = auto()
    USE_ENERGY = auto()
    FIXED_WEIGHTED = auto()


class LegacyGammaReconciliationWarning(FutureWarning):
    """A saved configuration or study selected retired reconciliation settings."""


LEGACY_KEY = "legacy_study_gamma_reconciliation"
DEFAULTS = {
    "method": "DISABLED",
    "low_beta_threshold": 0.9,
    "high_beta_threshold": 0.99,
    "low_beta_weight": 0.8,
    "high_beta_weight": 0.2,
    "mid_beta_weight": 0.5,
    "fixed_weight": 0.5,
}
PREFIX = "self_consistency_gamma_reconciliation_"


def normalize_legacy_gamma_payload(payload, *, warn=True):
    """Move saved flat keys into the study namespace; explicit nested keys win."""
    result = dict(payload)
    nested = result.get(LEGACY_KEY)
    if nested is not None and not isinstance(nested, Mapping):
        raise ValueError(f"{LEGACY_KEY} must be a mapping")
    legacy = {}
    for suffix in DEFAULTS:
        key = PREFIX + suffix
        if key in result:
            legacy[suffix] = result.pop(key)
    if nested is not None:
        unknown = set(nested) - DEFAULTS.keys()
        if unknown:
            raise ValueError(
                f"Unknown legacy gamma-reconciliation keys: {sorted(unknown)}"
            )
        legacy.update(nested)
    for suffix, value in legacy.items():
        if suffix != "method":
            legacy[suffix] = float(value)
    if legacy or nested is not None:
        result[LEGACY_KEY] = legacy
        if warn:
            warnings.warn(
                "Loading deprecated gamma-reconciliation study settings; "
                "use legacy_study_gamma_reconciliation for legacy interpretation.",
                LegacyGammaReconciliationWarning,
                stacklevel=2,
            )
    return result


def legacy_gamma_constructor(*, core=False):
    """Keep explicit old study kwargs without exposing them as dataclass fields."""
    prefix = "gamma_reconciliation_" if core else PREFIX

    def decorate(cls):
        original = cls.__init__

        @wraps(original)
        def initialize(self, *args, **kwargs):
            aliases = {
                suffix: kwargs.pop(prefix + suffix)
                for suffix in DEFAULTS
                if prefix + suffix in kwargs
            }
            nested = kwargs.get(LEGACY_KEY)
            if aliases:
                kwargs[LEGACY_KEY] = {**aliases, **(nested or {})}
            original(self, *args, **kwargs)
            if self.legacy_study_gamma_reconciliation is not None:
                # Validate the namespace even for direct study constructors.
                normalized = normalize_legacy_gamma_payload(
                    {LEGACY_KEY: self.legacy_study_gamma_reconciliation}, warn=False
                )
                self.legacy_study_gamma_reconciliation = normalized[LEGACY_KEY]
                warnings.warn(
                    "Gamma reconciliation is deprecated and retained only for "
                    "legacy studies. Use legacy_study_gamma_reconciliation to "
                    "interpret saved settings; ordinary runs use on-shell momentum.",
                    LegacyGammaReconciliationWarning,
                    stacklevel=2,
                )

        cls.__init__ = initialize
        for suffix, default in DEFAULTS.items():
            setattr(cls, prefix + suffix, _LegacySetting(suffix, default, core))
        return cls

    return decorate


class _LegacySetting:
    """Compatibility attribute for study code, outside the ordinary schema."""

    def __init__(self, suffix, default, core):
        self.suffix, self.default, self.core = suffix, default, core

    def __get__(self, instance, owner=None):
        values = getattr(instance, LEGACY_KEY, None) if instance is not None else None
        value = (values or {}).get(self.suffix, self.default)
        if self.suffix == "method":
            name = (
                value.name
                if isinstance(value, GammaReconciliationMethod)
                else str(value).upper()
            )
            if self.core:
                return GammaReconciliationMethod[name]
            return name
        return float(value)

    def __set__(self, instance, value):
        warnings.warn(
            "Flat gamma-reconciliation attributes are deprecated; use "
            "legacy_study_gamma_reconciliation for legacy studies.",
            LegacyGammaReconciliationWarning,
            stacklevel=2,
        )
        values = dict(getattr(instance, LEGACY_KEY, None) or {})
        values[self.suffix] = value
        setattr(instance, LEGACY_KEY, values)


def legacy_gamma_settings(config):
    """Read the explicit study namespace or old attributes on study objects."""
    if hasattr(config, LEGACY_KEY):
        values = getattr(config, LEGACY_KEY)
        return dict(values) if values is not None else None
    values = vars(config)
    old = {
        suffix: values[PREFIX + suffix]
        for suffix in DEFAULTS
        if PREFIX + suffix in values
    }
    return old or None


def legacy_gamma_payload(config):
    """Serialize only an explicitly selected legacy study, using JSON values."""
    values = getattr(config, LEGACY_KEY, None)
    if values is None:
        return {}
    return {
        LEGACY_KEY: {
            key: value.name if isinstance(value, GammaReconciliationMethod) else value
            for key, value in values.items()
        }
    }


def reconcile_legacy_gamma(config, gamma_from_energy, gamma_from_velocity, beta_total):
    """Preserve historical iteration-seed arithmetic without changing stored state."""
    method = config.gamma_reconciliation_method
    if method == GammaReconciliationMethod.USE_VELOCITY:
        gamma_reconciled = gamma_from_velocity
        alpha = 0.0
    elif method == GammaReconciliationMethod.USE_ENERGY:
        gamma_reconciled = gamma_from_energy
        alpha = 1.0
    elif method == GammaReconciliationMethod.FIXED_WEIGHTED:
        alpha = config.gamma_reconciliation_fixed_weight
        gamma_reconciled = (
            alpha * gamma_from_energy + (1.0 - alpha) * gamma_from_velocity
        )
    elif method == GammaReconciliationMethod.ADAPTIVE_WEIGHTED:
        low_threshold = config.gamma_reconciliation_low_beta_threshold
        high_threshold = config.gamma_reconciliation_high_beta_threshold
        low_weight = config.gamma_reconciliation_low_beta_weight
        high_weight = config.gamma_reconciliation_high_beta_weight
        mid_weight = config.gamma_reconciliation_mid_beta_weight
        if beta_total < low_threshold:
            alpha = low_weight
        elif beta_total > high_threshold:
            alpha = high_weight
        else:
            alpha = mid_weight
        gamma_reconciled = (
            alpha * gamma_from_energy + (1.0 - alpha) * gamma_from_velocity
        )
    else:
        gamma_reconciled = gamma_from_energy
        alpha = 1.0
    return gamma_reconciled, alpha
