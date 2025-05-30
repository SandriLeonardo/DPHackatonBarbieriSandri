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
        loss = (1 - (target_probs ** self.q)) / self.q
        return loss.mean()


class NCODLoss(nn.Module):
    """
    NCOD Loss function - placeholder for your teammate's GCOD implementation.
    You can replace this with the actual GCOD loss from their work.
    """

    def __init__(self):
        super(NCODLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Placeholder - replace with actual GCOD implementation
        return self.ce(logits, targets)


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


def get_loss_function(model_type, **kwargs):
    """
    Factory function to get the appropriate loss function.

    Args:
        model_type: str, one of ['gce', 'gcod', 'standard', 'noisy']
        **kwargs: additional arguments for loss functions

    Returns:
        Loss function instance
    """
    if model_type == "gce":
        q = kwargs.get('q', 0.5)  # Default q=0.5 from your config
        return GeneralizedCrossEntropyLoss(q=q)

    elif model_type == "gcod":
        return NCODLoss()

    elif model_type == "noisy":
        p_noisy = kwargs.get('p_noisy', 0.2)
        return NoisyCrossEntropyLoss(p_noisy=p_noisy)

    elif model_type == "standard":
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")