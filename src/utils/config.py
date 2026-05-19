from dataclasses import dataclass, field
from typing import List, Literal, Union

@dataclass
class PIDControlParams:
    k_p: float = 0.8
    dt: float = 0.1
    tau: float = 1.0
    alpha: float = 0.01

@dataclass
class BackpropControlParams:
    lr_c: float = 0.5
    momentum: float = 0.5

@dataclass
class PIDPlasticityParams:
    lr_w: float = 0.05

@dataclass
class BackpropPlasticityParams:
    lr_w: float = 0.5


@dataclass
class ExperimentConfig:
    # 1. High-Level Meta
    task: Literal["xor", "mnist"] = "xor"
    mode: Literal["backprop", "pid"] = "pid"
    seed: int = 7
    epochs: int = 1500
    max_steps: int = 50  # Shared by both optimization modes
    
    # 2. Network Anatomy
    pop_sizes: List[int] = field(default_factory=lambda: [2, 4, 1])
    dendritic_effect: Literal["additive", "multiplicative"] = "additive"
    leaky_slope: float = 0.01
    
    # 3. Mode-Dependent Parameters (Polymorphic)
    controller: Union[PIDControlParams, BackpropControlParams] = field(default_factory=PIDControlParams)
    plasticity: Union[PIDPlasticityParams, BackpropPlasticityParams] = field(default_factory=PIDPlasticityParams)

    def __post_init__(self):
        """Ensures the correct sub-configs match the selected mode."""
        if self.mode == "pid":
            if not isinstance(self.controller, PIDControlParams):
                self.controller = PIDControlParams()
            if not isinstance(self.plasticity, PIDPlasticityParams):
                self.plasticity = PIDPlasticityParams()
                
        elif self.mode == "backprop":
            if not isinstance(self.controller, BackpropControlParams):
                self.controller = BackpropControlParams()
            if not isinstance(self.plasticity, BackpropPlasticityParams):
                self.plasticity = BackpropPlasticityParams()