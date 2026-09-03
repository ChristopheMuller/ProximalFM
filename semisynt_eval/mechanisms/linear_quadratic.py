"""Closed-form SCM with randomly allocated linear and quadratic terms."""

from semisynt_eval.mechanisms.linear import LinearMechanism


class LinearQuadraticMechanism(LinearMechanism):
    """Use ``x`` or ``x²`` independently for every regression coefficient."""

    def _power_probabilities(self) -> tuple[float, float, float, float]:
        powers = self.config.powers
        return (
            powers.proxy_quadratic_probability,
            powers.treatment_quadratic_probability,
            powers.outcome_quadratic_probability,
            powers.effect_quadratic_probability,
        )
