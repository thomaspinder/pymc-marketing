#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Vector autoregression (VAR) baseline support, built on Impulso.

`Impulso <https://github.com/QuantClimate/Impulso>`_ is an optional dependency of
PyMC-Marketing, and the VAR baseline needs ``impulso>=0.1.3``. Install it with
``pip install 'pymc-marketing[var]'``. This module imports Impulso only when it is
used, so PyMC-Marketing imports without it.
"""

from importlib import metadata
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from packaging.version import Version
from pydantic import Field, InstanceOf
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import Model, MuEffect

if TYPE_CHECKING:
    from impulso import VAR

# Keep in step with the impulso pins in pyproject.toml.
_MIN_IMPULSO = "0.1.3"


def _import_impulso() -> ModuleType:
    """Import Impulso, the optional dependency of the VAR baseline.

    Impulso's version is read from its installed distribution. An Impulso without
    one, such as a source checkout put on ``sys.path``, is used unchecked, since
    its version cannot be known.

    Returns
    -------
    ModuleType
        The ``impulso`` module.

    Raises
    ------
    ImportError
        If Impulso is not installed, or is older than the minimum version.
    """
    try:
        import impulso
    except ImportError as exc:
        raise ImportError(
            "impulso is required for the VAR baseline. "
            "Install it with: pip install 'pymc-marketing[var]'"
        ) from exc
    try:
        installed = metadata.version("impulso")
    except metadata.PackageNotFoundError:
        return impulso
    if Version(installed) < Version(_MIN_IMPULSO):
        raise ImportError(
            f"impulso>={_MIN_IMPULSO} is required for the VAR baseline, but "
            f"impulso {installed} is installed. "
            "Upgrade it with: pip install -U 'pymc-marketing[var]'"
        )
    return impulso


class VARBaselineEffect(MuEffect):
    """Latent sales baseline that moves with brand metrics through a Bayesian VAR.

    Brand marketing builds mindset metrics such as awareness and consideration, and
    those metrics move the sales baseline over the long term [1]_. This effect adds
    that baseline to the MMM's mean as a latent series and fits it jointly with the
    rest of the MMM. Its prior is a vector autoregression (VAR) over the baseline and
    the brand metrics, with inputs such as brand spend as exogenous regressors, built
    by Impulso's ``VAR.build_in_model``. The
    :ref:`brand metrics notebook <mmm_brand_metrics_long_term>` fits the same model
    in two stages: an MMM whose time-varying intercept estimates the baseline, then a
    VAR on that estimate.

    The VAR runs on the baseline in the target's original units and on the brand
    columns in their own units, centered on their means over the MMM's dates. The
    baseline equation has no intercept, so the baseline is a zero-mean deviation
    around the MMM's intercept. Its own first-lag coefficient has prior mean
    ``baseline_own_lag_mean`` rather than the Minnesota prior's random walk.

    Parameters
    ----------
    brand_data : pd.DataFrame
        The brand metrics and exogenous inputs, one row per MMM date and in date
        order. Rows are matched to the MMM's dates by position.
    baseline_name : str
        Name of the latent baseline in the VAR. It must not be a column of
        ``brand_data``.
    endog_names : list[str]
        The VAR's endogenous series in order: ``baseline_name`` first, then columns
        of ``brand_data``.
    exog_names : list[str]
        Columns of ``brand_data`` that enter the VAR as exogenous regressors. May be
        empty.
    prefix : str
        Prefix for the effect's variables. The VAR's variables are named
        ``{prefix}::B``, ``{prefix}::latent`` and so on, and the baseline's
        contribution is ``{prefix}_effect_contribution``.
    var : impulso.VAR
        The VAR specification, with an integer ``lags``, constant volatility and
        Gaussian errors. It is not modified.
    baseline_own_lag_mean : float, default 0.0
        Prior mean of the baseline's own first-lag coefficient, strictly between -1
        and 1 because the baseline is a stationary deviation. It replaces the
        baseline's entry of a Minnesota prior's ``own_lag_mean``; the brand metrics
        keep theirs. It has no effect with any other prior.

    Notes
    -----
    The effect is for measurement only. Fitting and the contribution decomposition on
    the MMM's dates are supported; prediction on new dates and budget optimization
    are not.

    The brand metrics' likelihood and the constraint that keeps the baseline
    stationary are potentials, which prior and posterior predictive sampling ignore.
    Prior predictive baseline paths therefore come from the untruncated prior and can
    explode, and the brand metrics are never drawn.

    Sampling is tested with PyMC's NUTS sampler and with nutpie only. The
    stationarity constraint is untested on JAX, so ``nuts_sampler="numpyro"`` and
    ``nuts_sampler="blackjax"`` are unsupported.

    Impulso's coordinates, such as ``var``, ``coeff`` and ``exog``, are not
    prefixed, so every ``VARBaselineEffect`` in one MMM must name the same series.
    Impulso also adds a positional ``time`` coordinate that no variable uses.

    References
    ----------
    .. [1] Cain, P. M. (2022). "Modelling short- and long-term marketing effects in
       the consumer purchase journey." *International Journal of Research in
       Marketing*, 39(1), 96-116. https://doi.org/10.1016/j.ijresmar.2021.06.006

    Examples
    --------
    .. code-block:: python

        from impulso import VAR

        from pymc_marketing.mmm import (
            MMM,
            GeometricAdstock,
            LogisticSaturation,
            VARBaselineEffect,
        )

        brand_var = VARBaselineEffect(
            brand_data=df[["date", "awareness", "consideration", "brand_spend"]],
            baseline_name="baseline",
            endog_names=["baseline", "awareness", "consideration"],
            exog_names=["brand_spend"],
            prefix="brand_var",
            var=VAR(lags=1),
        )
        mmm = MMM(
            date_column="date",
            target_column="y",
            channel_columns=["x1", "x2"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
        ).add_mu_effect(brand_var)
        mmm.fit(df[["date", "x1", "x2"]], df["y"])

        # The baseline, in the target's units, next to the other components
        mmm.compute_mean_contributions_over_time()["brand_var_effect"]
    """

    brand_data: InstanceOf[pd.DataFrame]
    baseline_name: str
    endog_names: list[str]
    exog_names: list[str]
    prefix: str
    # Impulso is optional, so the spec's type is checked in `model_post_init`.
    var: Any
    baseline_own_lag_mean: float = Field(0.0, gt=-1, lt=1)

    def model_post_init(self, context: Any, /) -> None:
        """Check that ``var`` is an Impulso ``VAR`` specification."""
        impulso = _import_impulso()
        if not isinstance(self.var, impulso.VAR):
            raise TypeError(
                "var must be an impulso.VAR specification, "
                f"got {type(self.var).__name__}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict, without ``brand_data``."""
        return self.model_dump(mode="json", exclude={"brand_data"})

    def create_data(self, mmm: Model) -> None:
        """Check that ``brand_data`` has one row per MMM date.

        The brand data enter the graph as constants in :meth:`create_effect`, so no
        data variable is registered.

        Parameters
        ----------
        mmm : Model
            The MMM model instance.

        Raises
        ------
        ValueError
            If ``brand_data`` and the MMM have different numbers of dates.
        """
        n_dates = len(mmm.model.coords["date"])
        if len(self.brand_data) != n_dates:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} matches brand_data to the MMM's "
                f"dates by position, so it needs one row per date: got "
                f"{len(self.brand_data)} rows of brand_data for {n_dates} MMM dates."
            )

    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Build the VAR and return the baseline's contribution.

        The VAR is registered in a model named ``prefix`` nested in the MMM's model,
        which prefixes its variables.

        Parameters
        ----------
        mmm : Model
            The MMM model instance.

        Returns
        -------
        XTensorVariable
            The baseline in scaled-target units, with dims ``("date",)``.

        Raises
        ------
        ValueError
            If the target's AR(1) residual standard deviation is not positive, as for
            the all-zero target the MMM builds on when no ``y`` is given.
        """
        impulso = _import_impulso()
        observed = self.brand_data[self._observed_names].to_numpy(dtype=float)
        exog = None
        if self.exog_names:
            exog = self.brand_data[self.exog_names].to_numpy(dtype=float)
            exog = exog - exog.mean(axis=0)

        target = mmm.xarray_dataset["_target"].to_numpy()
        (baseline_scale,) = impulso.ar1_residual_sd(target[:, None])
        if not baseline_scale > 0:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs a target that varies: the "
                "AR(1) residual standard deviation of the target sets the scale of "
                f"the baseline, and it is {baseline_scale}. The MMM builds on an "
                "all-zero target when no y is given, as in sample_prior_predictive; "
                "pass y."
            )

        target_scale = mmm.model["target_scale"]
        spec = self._build_spec()
        with pm.Model(name=self.prefix):
            handles = spec.build_in_model(
                endog=observed - observed.mean(axis=0),
                exog=exog,
                n_lags=spec.lags,
                endog_names=self.endog_names,
                exog_names=self.exog_names or None,
                endog_scales=[baseline_scale, *impulso.ar1_residual_sd(observed)],
                intercept_equations=self._observed_names,
                latent_names=[self.baseline_name],
                latent_init_sigma=float(target_scale.get_value()),
            )

        baseline = ptx.as_xtensor(handles.latent[:, 0], dims=("date",))
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution", baseline / target_scale
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Do nothing: the brand data enter the graph as constants."""

    @property
    def _observed_names(self) -> list[str]:
        """The endogenous series observed in ``brand_data``: all but the baseline."""
        return self.endog_names[1:]

    def _build_spec(self) -> "VAR":
        """Return ``var`` with the baseline's own-lag prior mean set.

        Only a ``MinnesotaPrior`` has an own-lag mean, so any other prior is used
        unchanged.
        """
        impulso = _import_impulso()
        prior = self.var.resolved_prior
        if not isinstance(prior, impulso.MinnesotaPrior):
            return self.var

        own_lag_mean = prior.own_lag_mean
        if isinstance(own_lag_mean, tuple):
            observed_means = own_lag_mean[1:]
        else:
            observed_means = (own_lag_mean,) * len(self._observed_names)
        prior = prior.model_copy(
            update={"own_lag_mean": (self.baseline_own_lag_mean, *observed_means)}
        )
        return self.var.model_copy(update={"prior": prior})
