import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCrossEntropyLoss(nn.Module):
    """
    Generalized Cross Entropy Loss.
    """

    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        if not (0 < q <= 1):
            raise ValueError("q should be in (0, 1]")
        self.q = q

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        target_probs = probs[torch.arange(targets.size(0)), targets]
        target_probs = target_probs.clamp(min=1e-7, max=1 - 1e-7)
        loss = (1 - (target_probs ** self.q)) / self.q
        return loss.mean()


class GCODLoss(nn.Module):
    """
    Graph Centroid Outlier Discounting (GCOD) Loss Function
    Based on the NCOD method adapted for graph classification.
    The model parameters (theta) are updated using L1 + L3.
    The sample-specific parameters (u) are updated using L2.
    """

    def __init__(self, num_classes, alpha_train=0.01, lambda_r=0.1):
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha_train = alpha_train
        self.lambda_r = lambda_r
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def _ensure_u_shape(self, u_params, batch_size, target_ndim):
        """Helper to ensure u_params has the correct shape for operations."""
        if u_params.shape[0] != batch_size:
            raise ValueError(
                f"u_params batch dimension {u_params.shape[0]} does not match expected batch_size {batch_size}")

        if target_ndim == 1:
            return u_params.squeeze() if u_params.ndim > 1 else u_params
        elif target_ndim == 2:
            return u_params.unsqueeze(1) if u_params.ndim == 1 else u_params
        return u_params

    def compute_L1(self, logits, targets, u_params):
        """
        Computes L1 = CE(f_θ(Z_B)) + α_train * u_B * (y_B ⋅ ỹ_B)
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)
        ce_loss_values = self.ce_loss(logits, targets)
        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)
        feedback_term_values = self.alpha_train * current_u_params * (y_onehot * y_soft).sum(dim=1)
        L1 = ce_loss_values + feedback_term_values
        return L1.mean()

    def compute_L2(self, logits, targets, u_params):
        """
        Computes L2 = (1/|C|) * ||ỹ_B + u_B * y_B - y_B||²_F + λ_r * ||u_B||²_2
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)
        current_u_params_unsqueezed = self._ensure_u_shape(u_params, batch_size, target_ndim=2)
        term = y_soft + current_u_params_unsqueezed * y_onehot - y_onehot
        L2_reconstruction = (1.0 / self.num_classes) * torch.norm(term, p='fro').pow(2)
        current_u_params_1d = self._ensure_u_shape(u_params, batch_size, target_ndim=1)
        u_reg = self.lambda_r * torch.norm(current_u_params_1d, p=2).pow(2)
        L2 = L2_reconstruction + u_reg
        return L2

    def compute_L3(self, logits, targets, u_params, l3_coeff):
        """
        Computes L3 = l3_coeff * D_KL(L || σ(-log(u_B)))
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        diag_elements = (logits * y_onehot).sum(dim=1)
        L_log_probs = F.logsigmoid(diag_elements)
        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)
        target_probs_for_kl = torch.sigmoid(-torch.log(current_u_params + 1e-8))
        kl_div = F.kl_div(L_log_probs, target_probs_for_kl, reduction='mean', log_target=False)
        L3 = l3_coeff * kl_div
        return L3

    def forward(self, logits, targets, u_params, training_accuracy):
        """
        Calculates the GCOD loss components.
        Returns: (total_loss_for_theta, L1, L2, L3)
        """
        calculated_L1 = self.compute_L1(logits, targets, u_params)
        calculated_L2 = self.compute_L2(logits, targets, u_params)
        l3_coefficient = (1.0 - training_accuracy)
        calculated_L3 = self.compute_L3(logits, targets, u_params, l3_coefficient)
        total_loss_for_theta = calculated_L1 + calculated_L3
        return total_loss_for_theta, calculated_L1, calculated_L2, calculated_L3


class NoisyCrossEntropyLoss(nn.Module):
    """
    Noisy Cross Entropy Loss.
    """

    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - F.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()


def get_loss_function(loss_type, **kwargs):
    """
    Factory function to get the appropriate loss function.
    """
    if loss_type == "gce":
        q = kwargs.get('q', 0.5)
        return GeneralizedCrossEntropyLoss(q=q)

    elif loss_type == "gcod":
        num_classes = kwargs.get('num_classes', 6)
        alpha_train = kwargs.get('alpha_train', 0.01)
        lambda_r = kwargs.get('lambda_r', 0.1)
        return GCODLoss(num_classes=num_classes, alpha_train=alpha_train, lambda_r=lambda_r)

    elif loss_type == "noisy":
        p_noisy = kwargs.get('p_noisy', 0.2)
        return NoisyCrossEntropyLoss(p_noisy=p_noisy)

    elif loss_type == "standard":
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")