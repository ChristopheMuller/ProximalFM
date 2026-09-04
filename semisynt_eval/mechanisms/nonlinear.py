"""Mixed normalized nonlinear proximal SCM."""

from semisynt_eval.mechanisms.linear import BasisMechanism


class NonlinearMechanism(BasisMechanism):
    family = "nonlinear"
