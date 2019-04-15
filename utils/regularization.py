class EarlyStopping:
    def __init__(self, tolerance, patience):
        """
        Args:
          patience (int): Maximum number of epochs with unsuccessful updates.
          tolerance (int): We assume that the update is unsuccessful if the validation error is larger
                            than the best validation error so far plus this tolerance.
        """
        self.tolerance = tolerance
        self.patience = patience

    def stop_criterion(self, val_errors):
        """
        Args:
          val_errors (iterable): Validation errors after every update during training.

        Returns: True if training should be stopped: when the validation error is larger than the best
                  validation error obtained so far (with given tolearance) for patience epochs.
                 Otherwise returns False.
        """
        min_val = float("inf")
        num_unsuccess = 0
        for error in val_errors:
            if error < min_val:
                min_val = error
                num_unsuccess = 0
            elif error - min_val > self.tolerance:
                num_unsuccess += 1
                if num_unsuccess >= self.tolerance:
                    return True
        return False