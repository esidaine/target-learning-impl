from dataclasses import dataclass, field
from typing import List, Literal, Union, Optional


@dataclass
class PIControlParams:
    """PIControlParams."""

    k_p: float = 0.8
    dt: float = 0.1
    tau: float = 1.0
    alpha: float = 0.01
    max_steps: int = 10
    use_derivative: bool = True


@dataclass
class BackpropControlParams:
    """BackpropControlParams."""

    lr_c: float = 0.5
    momentum: float = 0.5
    max_steps: int = 100


@dataclass
class PIPlasticityParams:
    """PIPlasticityParams."""

    lr_w: float = 0.0001  # CHANGE TO 0.0001 for mnist, 0.05 for xor


@dataclass
class BackpropPlasticityParams:
    """BackpropPlasticityParams."""

    lr_w: float = 0.5


@dataclass
class ExperimentConfig:
    """ExperimentConfig."""

    # 1. High-Level Meta
    task: Literal["xor", "mnist"] = "xor"
    mode: Literal["backprop", "pi"] = "pi"
    seed: int = 7
    epochs: int = 3  # CHANGE for real training to 800

    # 2. Network Anatomy
    pop_sizes: Optional[List[int]] = None
    dendritic_effect: Literal["additive", "multiplicative"] = "additive"
    leaky_slope: float = 0.01

    # 3. Mode-Dependent Parameters (Polymorphic)
    controller: Union[PIControlParams, BackpropControlParams] = field(
        default_factory=PIControlParams
    )
    plasticity: Union[PIPlasticityParams, BackpropPlasticityParams] = field(
        default_factory=PIPlasticityParams
    )

    def __post_init__(self):
        """Fill architecture defaults and validate mode-specific parameters.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If a controller or plasticity config mismatches the mode.
        """
        if self.pop_sizes is None:
            self.pop_sizes = (
                [2, 8, 1] if self.task == "xor" else [784, 256, 128, 64, 10]
            )

        if self.mode == "pi":
            if not isinstance(self.controller, PIControlParams):
                raise ValueError(
                    f"Mismatch: mode is 'pi', but controller is {type(self.controller).__name__}"
                )
            if not isinstance(self.plasticity, PIPlasticityParams):
                raise ValueError(
                    f"Mismatch: mode is 'pi', but plasticity is {type(self.plasticity).__name__}"
                )

        elif self.mode == "backprop":
            if not isinstance(self.controller, BackpropControlParams):
                raise ValueError(
                    f"Mismatch: mode is 'backprop', but controller is {type(self.controller).__name__}"
                )
            if not isinstance(self.plasticity, BackpropPlasticityParams):
                raise ValueError(
                    f"Mismatch: mode is 'backprop', but plasticity is {type(self.plasticity).__name__}"
                )
