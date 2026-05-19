from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from core.trainer import Trainer
import torch
import pytest

@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
@pytest.mark.parametrize("mode", ["backprop", "pid"])
def test_single_step_loss_decreases(tiny_network, tiny_batch, mode, dendritic_effect): 
    tiny_network.dendritic_effect = dendritic_effect
    
    controller = ControlMechanism(mode=mode)
    plasticity = Plasticity(lr_w=0.5) 
    trainer = Trainer(tiny_network, controller, plasticity)

    x,_ = tiny_batch

    # loss BEFORE: forward with old weights
    with torch.no_grad():
        y_init = tiny_network(x, control_signals=None, save_baseline=False)
        random_target = y_init + 0.2 * torch.randn_like(y_init)
        init_loss = trainer.criterion(y_init, random_target).item()


    # -------------------------------------------------------------------------    
    # Find the control signal that should yield an improvement over the baseline, 
    # and update the weights based on that control signal.
    # -------------------------------------------------------------------------

    trainer.controller.optimize_control_signal(
                sensory_inputs=x, 
                target_y=random_target, 
                network=trainer.network
            ) 

    plasticity.update_weights(
                network=trainer.network, 
                sensory_inputs=x
            )

    # Get NEW loss: forward with NEW weights, same x, same target
    with torch.no_grad():
        y_new = tiny_network(x, control_signals=None, save_baseline=False)
        post_update_loss = trainer.criterion(y_new, random_target).item()
    
    
    assert post_update_loss < init_loss, (
        f"[{mode.upper()}] A single weight update failed to improve baseline loss. "
        f"Initial: {init_loss:.6f} -> Post-Update: {post_update_loss:.6f}"
    )

    


