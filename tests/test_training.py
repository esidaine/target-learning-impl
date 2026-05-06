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
    losses = []

    # loss BEFORE: forward with old weights
    with torch.no_grad():
        y_init = tiny_network(x, control_signals=None, save_baseline=False)
        losses.append(trainer.criterion(y_init, random_target).item())
    for _ in range(3):
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
        
        losses.append(trainer.criterion(y_new, random_target).item())

    # Assert strict monotonic decrease
    # losses will have 4 items: [init_loss, loss_step1, loss_step2, loss_step3]
    for i in range(len(losses) - 1):
        assert losses[i] > losses[i+1], (
            f"Loss failed to strictly decrease from step {i} to {i+1}. "
            f"Losses: {losses[i]:.6f} -> {losses[i+1]:.6f}"
        )

    


