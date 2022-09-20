import torch
import numpy as np
from detectron2.modeling.matcher import Matcher as D2Matcher


class Matcher(D2Matcher):

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of the
        Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.

        Note - this is a batched version, aimed at reducing memory footprint that causes
               failures in the original `detectron2` implementation
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        # @Tsiper - Batch based implementation
        num_gt_indices = highest_quality_foreach_gt.shape[0]
        split_ind = 50
        num_of_batches = np.ceil(num_gt_indices / split_ind).astype(np.int)
        for ind in range(num_of_batches):
            ind_start = int(ind * split_ind)
            ind_end = min(int((ind + 1) * split_ind), num_gt_indices)
            full_mat_matches = match_quality_matrix[ind_start: ind_end, :] == highest_quality_foreach_gt[
                                                                              ind_start: ind_end, None]
            gt_pred_pairs_of_highest_quality = torch.nonzero(torch.as_tensor(full_mat_matches))

            pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
            match_labels[pred_inds_to_update] = 1

    def __call__(self, match_quality_matrix):
        """
        In this local override, we remove erroneous assertions that interfere with rotated boxes
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        # # @tsiper: Original matcher contained this assertion. We replace with the nullification of negative values
        # assert torch.all(match_quality_matrix >= 0)
        match_quality_matrix = torch.nn.functional.relu_(match_quality_matrix)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

