import torch
from opacus.optimizers import DPOptimizer


class DiNNODPOptimizer(DPOptimizer):

    def clear_pred_grads(self):
        """
        Save prediction gradients before computing remaining gradients
        """
        self.pred_loss_grads = []
        with torch.no_grad():
            for p in self.params:
                p_grad = p.grad.detach().clone()
                self.pred_loss_grads.append(p_grad)
        self.zero_grad()

    def dinno_step(self):
        """
        Merge prediction loss gradients with remaining gradients
        then call underlying ``optimizer.step()``
        """
        with torch.no_grad():
            for i, p in enumerate(self.params):
                p.grad += self.pred_loss_grads[i]
        return self.original_optimizer.step()