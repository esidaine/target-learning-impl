from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from core.trainer import Trainer
import torch


def test_single_step_loss_decreases(tiny_network, tiny_batch): 
    controller = ControlMechanism(mode='pid', lr_c=0.1, max_steps=60)
    plasticity = Plasticity(lr_theta=0.2) 
    trainer = Trainer(tiny_network, controller, plasticity)

    x,_ = tiny_batch
    out_dim = tiny_network.populations[-1].num_neurons
    random_target = torch.randn(x.size(0), out_dim)

    # loss BEFORE: forward with old weights
    with torch.no_grad():
        y_before = tiny_network(x, control_signals=None, save_baseline=False)

    loss_before = trainer.criterion(y_before, random_target)

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

    # loss AFTER: forward with NEW weights, same x, same target
    with torch.no_grad():
        y_after = tiny_network(x, control_signals=None, save_baseline=False)
    
    loss_after = trainer.criterion(y_after, random_target)

    assert loss_after < loss_before

    


