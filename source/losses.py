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


class GCODLoss_C(nn.Module):
    """
    Graph Centroid Outlier Discounting (GCOD) Loss Function
    Based on the NCOD method adapted for graph classification.
    The model parameters (theta) are updated using L1 + L3.
    The sample-specific parameters (u) are updated using L2.
    """
    def __init__(self, num_classes, alpha_train=5.0, lambda_r=0.5): # Added lambda_r
        """
        Args:
            num_classes (int): Number of classes.
            alpha_train (float): Corresponds to lambda_p in args, coefficient for the
                                 feedback term in L1.
            lambda_r (float): Coefficient for the u regularization term in L2.
        """
        super(GCODLoss_C, self).__init__()
        self.num_classes = num_classes
        self.alpha_train = alpha_train
        self.lambda_r = lambda_r # Store lambda_r
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') # for per-sample CE

    def _ensure_u_shape(self, u_params, batch_size, target_ndim):
        """Helper to ensure u_params has the correct shape for operations."""
        if u_params.shape[0] != batch_size:
            raise ValueError(f"u_params batch dimension {u_params.shape[0]} does not match expected batch_size {batch_size}")

        if target_ndim == 1: # Expected shape [batch_size]
            return u_params.squeeze() if u_params.ndim > 1 else u_params
        elif target_ndim == 2: # Expected shape [batch_size, 1]
            return u_params.unsqueeze(1) if u_params.ndim == 1 else u_params
        return u_params


    def compute_L1(self, logits, targets, u_params):
        """
        Computes L1 = CE(f_θ(Z_B)) + α_train * u_B * (y_B ⋅ ỹ_B)
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
        Returns:
            Tensor: Scalar L1 loss for the batch.
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)

        ce_loss_values = self.ce_loss(logits, targets) # Shape: [batch_size]

        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)

        feedback_term_values = self.alpha_train * current_u_params * (y_onehot * y_soft).sum(dim=1) # Shape: [batch_size]

        L1 = ce_loss_values + feedback_term_values
        return L1.mean()

    def compute_L2(self, logits, targets, u_params):
        """
        Computes L2 = (1/|C|) * ||ỹ_B + u_B * y_B - y_B||²_F + λ_r * ||u_B||²_2
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
        Returns:
            Tensor: Scalar L2 loss for the batch (for u optimization).
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)

        current_u_params_unsqueezed = self._ensure_u_shape(u_params, batch_size, target_ndim=2)

        term = y_soft + current_u_params_unsqueezed * y_onehot - y_onehot # Shape: [batch_size, num_classes]

        # L2 reconstruction term (Frobenius norm for matrix part)
        L2_reconstruction = (1.0 / self.num_classes) * torch.norm(term, p='fro').pow(2)
        
        # u regularization term (L2 norm for u_params vector part)
        # Ensure u_params is 1D for this norm calculation
        current_u_params_1d = self._ensure_u_shape(u_params, batch_size, target_ndim=1)
        u_reg = self.lambda_r * torch.norm(current_u_params_1d, p=2).pow(2)

        L2 = L2_reconstruction + u_reg
        return L2

    def compute_L3(self, logits, targets, u_params, l3_coeff):
        """
        Computes L3 = l3_coeff * D_KL(L || σ(-log(u_B)))
                     where l3_coeff = (1 - training_accuracy)
                     and L = log(σ(logit_true_class)) are log-probabilities
                     and σ(-log(u_B)) are probabilities
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
            l3_coeff (float): Coefficient for the KL divergence term, e.g., (1 - training_accuracy).
        Returns:
            Tensor: Scalar L3 loss for the batch.
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Logit of the true class for each sample in the batch
        diag_elements = (logits * y_onehot).sum(dim=1) # Shape: [batch_size]

        # L_log_probs = log(sigma(true_class_logit)) which are log-probabilities
        L_log_probs = F.logsigmoid(diag_elements) # Shape: [batch_size]

        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)

        # target_probs_for_kl = sigma(-log(u_B)) which are probabilities
        target_probs_for_kl = torch.sigmoid(-torch.log(current_u_params + 1e-8)) # Shape: [batch_size]

        # F.kl_div expects input (L_log_probs) as log-probabilities and target (target_probs_for_kl) as probabilities.
        # reduction='mean' averages the loss over all elements in the batch.
        # log_target=False means target_probs_for_kl are probabilities, not log-probabilities.
        kl_div = F.kl_div(L_log_probs, target_probs_for_kl, reduction='mean', log_target=False)

        L3 = l3_coeff * kl_div
        return L3

    def forward(self, logits, targets, u_params, training_accuracy):
        """
        Calculates the GCOD loss components.
        The main loss for model (theta) update is L1 + L3.
        L2 is primarily used for updating u_params (called separately).
        Args:
            logits (Tensor): Model output logits.
            targets (Tensor): Ground truth labels.
            u_params (Tensor): Per-sample u values for the batch.
            training_accuracy (float): The actual training accuracy (value between 0 and 1)
                                     for the current batch or epoch.
        Returns:
            tuple: (total_loss_for_theta, L1, L2, L3)
                   total_loss_for_theta = L1 + L3
        """
        calculated_L1 = self.compute_L1(logits, targets, u_params)
        # L2 is calculated here mainly for complete reporting if needed,
        # but the train loop will call compute_L2 separately for u-optimization.
        # This L2 will now include the regularization term.
        calculated_L2 = self.compute_L2(logits, targets, u_params)

        l3_coefficient = (1.0 - training_accuracy) # As per GCOD paper (1 - alpha_train where alpha_train is accuracy)
        calculated_L3 = self.compute_L3(logits, targets, u_params, l3_coefficient)

        total_loss_for_theta = calculated_L1 + calculated_L3

        return total_loss_for_theta, calculated_L1, calculated_L2, calculated_L3


class GCODLoss_D(nn.Module):
    """
    Graph Centroid Outlier Discounting (GCOD) Loss Function
    Based on the NCOD method adapted for graph classification.
    The model parameters (theta) are updated using L1 + L3.
    The sample-specific parameters (u) are updated using L2.
    """
    def __init__(self, num_classes, alpha_train=2.0):
        """
        Args:
            num_classes (int): Number of classes.
            alpha_train (float): Corresponds to lambda_p in args, coefficient for the
                                 feedback term in L1.
        """
        super(GCODLoss_D, self).__init__()
        self.num_classes = num_classes
        self.alpha_train = alpha_train
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') # for per-sample CE

    def _ensure_u_shape(self, u_params, batch_size, target_ndim):
        """Helper to ensure u_params has the correct shape for operations."""
        if u_params.shape[0] != batch_size:
            raise ValueError(f"u_params batch dimension {u_params.shape[0]} does not match expected batch_size {batch_size}")

        if target_ndim == 1: # Expected shape [batch_size]
            return u_params.squeeze() if u_params.ndim > 1 else u_params
        elif target_ndim == 2: # Expected shape [batch_size, 1]
            return u_params.unsqueeze(1) if u_params.ndim == 1 else u_params
        return u_params


    def compute_L1(self, logits, targets, u_params):
        """
        Computes L1 = CE(f_θ(Z_B)) + α_train * u_B * (y_B ⋅ ỹ_B)
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
        Returns:
            Tensor: Scalar L1 loss for the batch.
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)

        ce_loss_values = self.ce_loss(logits, targets) # Shape: [batch_size]

        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)

        feedback_term_values = self.alpha_train * current_u_params * (y_onehot * y_soft).sum(dim=1) # Shape: [batch_size]

        L1 = ce_loss_values + feedback_term_values
        return L1.mean()

    def compute_L2(self, logits, targets, u_params):
        """
        Computes L2 = (1/|C|) * ||ỹ_B + u_B * y_B - y_B||²
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
        Returns:
            Tensor: Scalar L2 loss for the batch.
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)

        current_u_params_unsqueezed = self._ensure_u_shape(u_params, batch_size, target_ndim=2)

        term = y_soft + current_u_params_unsqueezed * y_onehot - y_onehot # Shape: [batch_size, num_classes]

        # L2 norm squared of the matrix 'term', then scaled
        L2 = (1.0 / self.num_classes) * torch.norm(term, p='fro').pow(2) # Frobenius norm for matrix
        return L2

    def compute_L3(self, logits, targets, u_params, l3_coeff):
        """
        Computes L3 = l3_coeff * D_KL(L || σ(-log(u_B)))
                     where l3_coeff = (1 - training_accuracy)
                     and L = log(σ(logit_true_class)) are log-probabilities
                     and σ(-log(u_B)) are probabilities
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
            l3_coeff (float): Coefficient for the KL divergence term, e.g., (1 - training_accuracy).
        Returns:
            Tensor: Scalar L3 loss for the batch.
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Logit of the true class for each sample in the batch
        diag_elements = (logits * y_onehot).sum(dim=1) # Shape: [batch_size]

        # L_log_probs = log(sigma(true_class_logit)) which are log-probabilities
        L_log_probs = F.logsigmoid(diag_elements) # Shape: [batch_size]

        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)

        # target_probs_for_kl = sigma(-log(u_B)) which are probabilities
        target_probs_for_kl = torch.sigmoid(-torch.log(current_u_params + 1e-8)) # Shape: [batch_size]

        # F.kl_div expects input (L_log_probs) as log-probabilities and target (target_probs_for_kl) as probabilities.
        # reduction='mean' averages the loss over all elements in the batch.
        # log_target=False means target_probs_for_kl are probabilities, not log-probabilities.
        kl_div = F.kl_div(L_log_probs, target_probs_for_kl, reduction='mean', log_target=False)

        L3 = l3_coeff * kl_div
        return L3

    def forward(self, logits, targets, u_params, training_accuracy):
        """
        Calculates the GCOD loss components.
        The main loss for model (theta) update is L1 + L3.
        L2 is primarily used for updating u_params (called separately).
        Args:
            logits (Tensor): Model output logits.
            targets (Tensor): Ground truth labels.
            u_params (Tensor): Per-sample u values for the batch.
            training_accuracy (float): The actual training accuracy (value between 0 and 1)
                                     for the current batch or epoch.
        Returns:
            tuple: (total_loss_for_theta, L1, L2, L3)
                   total_loss_for_theta = L1 + L3
        """
        calculated_L1 = self.compute_L1(logits, targets, u_params)
        # L2 is calculated here mainly for complete reporting if needed,
        # but the train loop will call compute_L2 separately for u-optimization.
        calculated_L2 = self.compute_L2(logits, targets, u_params)

        l3_coefficient = (1.0 - training_accuracy) # As per GCOD paper (1 - alpha_train where alpha_train is accuracy)
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

    elif loss_type == "gcod_c":
        num_classes = kwargs.get('num_classes', 6)
        alpha_train = kwargs.get('alpha_train', 5.0)
        lambda_r = kwargs.get('lambda_r', 0.5)
        return GCODLoss_C(num_classes=num_classes, alpha_train=alpha_train, lambda_r=lambda_r)
    
    elif loss_type == "gcod_d":
        num_classes = kwargs.get('num_classes', 6)
        alpha_train = kwargs.get('alpha_train', 2.0)
        return GCODLoss_D(num_classes=num_classes, alpha_train=alpha_train)

    elif loss_type == "noisy":
        p_noisy = kwargs.get('p_noisy', 0.2)
        return NoisyCrossEntropyLoss(p_noisy=p_noisy)

    elif loss_type == "standard":
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")