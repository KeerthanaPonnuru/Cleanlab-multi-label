
# Copyright (C) 2017-2023  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.


import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple, Optional
from rank import get_label_quality_scores
from cleanlab.internal.util import get_num_classes, value_counts
from cleanlab.internal.constants import CLIPPING_LOWER_BOUND

from cleanlab.internal.multiannotator_utils import (
    assert_valid_inputs_multiannotator,
    assert_valid_pred_probs,
    check_consensus_label_classes,
    find_best_temp_scaler,
    temp_scale_pred_probs,
)


def get_label_quality_multiannotator(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: np.ndarray,
    *,
    consensus_method: Union[str, List[str]] = "majority_vote",
    quality_method: str = "crowdlab",
    calibrate_probs: bool = False,
    return_detailed_quality: bool = True,
    return_annotator_stats: bool = True,
    return_weights: bool = False,
    verbose: bool = True,
    label_quality_score_kwargs: dict = {},
) -> Dict[str, Any]:
    """Returns label quality scores for each example and for each annotator in a dataset labeled by multiple annotators.

    This function is for multiclass classification datasets where examples have been labeled by
    multiple annotators (not necessarily the same number of annotators per example).

    It computes one consensus label for each example that best accounts for the labels chosen by each
    annotator (and their quality), as well as a consensus quality score for how confident we are that this consensus label is actually correct.
    It also computes similar quality scores for each annotator's individual labels, and the quality of each annotator.
    Scores are between 0 and 1 (estimated via methods like CROWDLAB); lower scores indicate labels/annotators less likely to be correct.

    To decide what data to collect additional labels for, try the `~cleanlab.multiannotator.get_active_learning_scores`
    (ActiveLab) function, which is intended for active learning with multiple annotators.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        ``labels_multiannotator[n][m]`` = label for n-th example given by m-th annotator.

        For a dataset with K classes, each given label must be an integer in 0, 1, ..., K-1 or ``NaN`` if this annotator did not label a particular example.
        If you have string or other differently formatted labels, you can convert them to the proper format using :py:func:`format_multiannotator_labels <cleanlab.internal.multiannotator_utils.format_multiannotator_labels>`.
        If pd.DataFrame, column names should correspond to each annotator's ID.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    consensus_method : str or List[str], default = "majority_vote"
        Specifies the method used to aggregate labels from multiple annotators into a single consensus label.
        Options include:

        * ``majority_vote``: consensus obtained using a simple majority vote among annotators, with ties broken via ``pred_probs``.
        * ``best_quality``: consensus obtained by selecting the label with highest label quality (quality determined by method specified in ``quality_method``).

        A List may be passed if you want to consider multiple methods for producing consensus labels.
        If a List is passed, then the 0th element of the list is the method used to produce columns `consensus_label`, `consensus_quality_score`, `annotator_agreement` in the returned DataFrame.
        The remaning (1st, 2nd, 3rd, etc.) elements of this list are output as extra columns in the returned pandas DataFrame with names formatted as:
        `consensus_label_SUFFIX`, `consensus_quality_score_SUFFIX` where `SUFFIX` = each element of this
        list, which must correspond to a valid method for computing consensus labels.
    quality_method : str, default = "crowdlab"
        Specifies the method used to calculate the quality of the consensus label.
        Options include:

        * ``crowdlab``: an emsemble method that weighs both the annotators' labels as well as the model's prediction.
        * ``agreement``: the fraction of annotators that agree with the consensus label.
    calibrate_probs : bool, default = False
        Boolean value that specifies whether the provided `pred_probs` should be re-calibrated to better match the annotators' empirical label distribution.
        We recommend setting this to True in active learning applications, in order to prevent overconfident models from suggesting the wrong examples to collect labels for.
    return_detailed_quality: bool, default = True
        Boolean to specify if `detailed_label_quality` is returned.
    return_annotator_stats : bool, default = True
        Boolean to specify if `annotator_stats` is returned.
    return_weights : bool, default = False
        Boolean to specify if `model_weight` and `annotator_weight` is returned.
        Model and annotator weights are applicable for ``quality_method == crowdlab``, will return ``None`` for any other quality methods.
    verbose : bool, default = True
        Important warnings and other printed statements may be suppressed if ``verbose`` is set to ``False``.
    label_quality_score_kwargs : dict, optional
        Keyword arguments to pass into :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    Returns
    -------
    labels_info : dict
        Dictionary containing up to 5 pandas DataFrame with keys as below:

        ``label_quality`` : pandas.DataFrame
            pandas DataFrame in which each row corresponds to one example, with columns:

            * ``num_annotations``: the number of annotators that have labeled each example.
            * ``consensus_label``: the single label that is best for each example (you can control how it is derived from all annotators' labels via the argument: ``consensus_method``).
            * ``annotator_agreement``: the fraction of annotators that agree with the consensus label (only consider the annotators that labeled that particular example).
            * ``consensus_quality_score``: label quality score for consensus label, calculated by the method specified in ``quality_method``.

        ``detailed_label_quality`` : pandas.DataFrame
            Only returned if `return_detailed_quality=True`.
            Returns a pandas DataFrame with columns `quality_annotator_1`, `quality_annotator_2`, ..., `quality_annotator_M` where each entry is
            the label quality score for the labels provided by each annotator (is ``NaN`` for examples which this annotator did not label).

        ``annotator_stats`` : pandas.DataFrame
            Only returned if `return_annotator_stats=True`.
            Returns overall statistics about each annotator, sorted by lowest annotator_quality first.
            pandas DataFrame in which each row corresponds to one annotator (the row IDs correspond to annotator IDs), with columns:

            * ``annotator_quality``: overall quality of a given annotator's labels, calculated by the method specified in ``quality_method``.
            * ``num_examples_labeled``: number of examples annotated by a given annotator.
            * ``agreement_with_consensus``: fraction of examples where a given annotator agrees with the consensus label.
            * ``worst_class``: the class that is most frequently mislabeled by a given annotator.

        ``model_weight`` : float
            Only returned if `return_weights=True`. It is only applicable for ``quality_method == crowdlab``.
            The model weight specifies the weight of classifier model in weighted averages used to estimate label quality
            This number is an estimate of how trustworthy the model is relative the annotators.

        ``annotator_weight`` : np.ndarray
            Only returned if `return_weights=True`. It is only applicable for ``quality_method == crowdlab``.
            An array of shape ``(M,)`` where M is the number of annotators, specifying the weight of each annotator in weighted averages used to estimate label quality.
            These weights are estimates of how trustworthy each annotator is relative to the other annotators.

    """

    if isinstance(labels_multiannotator, pd.DataFrame):
        annotator_ids = labels_multiannotator.columns
        index_col = labels_multiannotator.index
        labels_multiannotator = (
            labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
        )
    elif isinstance(labels_multiannotator, np.ndarray):
        annotator_ids = None
        index_col = None
    else:
        raise ValueError("labels_multiannotator must be either a NumPy array or Pandas DataFrame.")

    if return_weights == True and quality_method != "crowdlab":
        raise ValueError(
            "Model and annotator weights are only applicable to the crowdlab quality method. "
            "Either set return_weights=False or quality_method='crowdlab'."
        )

    # Count number of non-NaN values for each example
    num_annotations=np.sum(~np.isnan(labels_multiannotator).any(axis=2), axis=1)


    # calibrate pred_probs
    if calibrate_probs:
        optimal_temp = find_best_temp_scaler(labels_multiannotator, pred_probs)
        pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)

    if not isinstance(consensus_method, list):
        consensus_method = [consensus_method]

    if "best_quality" in consensus_method or "majority_vote" in consensus_method:
        majority_vote_label = get_majority_vote_label(
            labels_multiannotator=labels_multiannotator,
            pred_probs=pred_probs,
            verbose=False,
        )

        (
            MV_annotator_agreement,
            MV_consensus_quality_score,
            MV_post_pred_probs,
            MV_model_weight,
            MV_annotator_weight,
        ) = _get_consensus_stats(
            labels_multiannotator=labels_multiannotator,
            pred_probs=pred_probs,
            num_annotations=num_annotations,
            consensus_label=majority_vote_label,
            quality_method=quality_method,
            verbose=verbose,
            label_quality_score_kwargs=label_quality_score_kwargs,
        )

    

    valid_methods = ["majority_vote", "best_quality"]
    main_method = True

    for curr_method in consensus_method:
        # geting consensus label and stats
        if curr_method == "majority_vote":
            consensus_label = majority_vote_label
            annotator_agreement = MV_annotator_agreement
            consensus_quality_score = MV_consensus_quality_score
            post_pred_probs = MV_post_pred_probs
            model_weight = MV_model_weight
            annotator_weight = MV_annotator_weight

        else:
            raise ValueError(
                f"""
                {curr_method} is not a valid consensus method!
                Please choose a valid consensus_method: {valid_methods}
                """
            )

        if main_method:
            #print('mw',model_weight)
            #print('aw',type(annotator_weight))
            #print('aw',annotator_weight.shape)

            
            annotators=['wkratochvil', 'kmaisuria', 'hannahweisman', 'ezequielbautista','aribahali', 'kaelynnrodriguez', 'babatundeshofolu','rishika.patel@ufl.edu\\health', 'jennifer.noa']
            
            label_quality = pd.DataFrame({
                'num_annotations': num_annotations,
                'consensus_label': consensus_label.tolist(),
                'annotator_agreement': annotator_agreement.tolist(),
                'consensus_quality_score': consensus_quality_score.tolist()
            })
            
            
            detailed_label_quality = []

            for m in range(labels_multiannotator.shape[1]):  
                annotator_labels = labels_multiannotator[:, m, :]
                
                annotator_quality_scores = _get_annotator_label_quality_score(
                    annotator_labels, 
                    post_pred_probs, 
                    label_quality_score_kwargs
                )
                
                detailed_label_quality.append(annotator_quality_scores)

            detailed_label_quality=np.array(detailed_label_quality)
            detailed_label_quality = detailed_label_quality.T

            df = pd.DataFrame(detailed_label_quality, columns=[f'Quality_{i}' for i in annotators])

            annotator_stats = _get_annotator_stats(
                    labels_multiannotator=labels_multiannotator,
                    pred_probs=post_pred_probs,
                    consensus_label=consensus_label,
                    num_annotations=num_annotations,
                    annotator_agreement=annotator_agreement,
                    model_weight=model_weight,
                    annotator_weight=annotator_weight,
                    consensus_quality_score=consensus_quality_score,
                    detailed_label_quality=detailed_label_quality,
                    annotator_ids=annotator_ids,
                    quality_method=quality_method,
                )
            annotator_weight= pd.DataFrame(annotator_weight)
            
                
    labels_info = {
        "label_quality": label_quality,
        "detailed_label_quality":df,
        "annotator_stats":annotator_stats,
        "model_weight" : model_weight[0],
        "annotator_weight" : annotator_weight
    }

    
    return labels_info 


def get_label_quality_multiannotator_ensemble(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: np.ndarray,
    *,
    calibrate_probs: bool = False,
    return_detailed_quality: bool = True,
    return_annotator_stats: bool = True,
    return_weights: bool = False,
    verbose: bool = True,
    label_quality_score_kwargs: dict = {},
) -> Dict[str, Any]:
    """Returns label quality scores for each example and for each annotator, based on predictions from an ensemble of models.

    This function is similar to `~cleanlab.multiannotator.get_label_quality_multiannotator` but for settings where
    you have trained an ensemble of multiple classifier models rather than a single model.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        Multiannotator labels in the same format expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    pred_probs : np.ndarray
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    calibrate_probs : bool, default = False
        Boolean value as expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    return_detailed_quality: bool, default = True
        Boolean value as expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    return_annotator_stats : bool, default = True
        Boolean value as expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    return_weights : bool, default = False
        Boolean value as expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    verbose : bool, default = True
        Boolean value as expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    label_quality_score_kwargs : dict, optional
        Keyword arguments in the same format expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.

    Returns
    -------
    labels_info : dict
        Dictionary containing up to 5 pandas DataFrame with keys as below:

        ``label_quality`` : pandas.DataFrame
            Similar to output as `~cleanlab.multiannotator.get_label_quality_multiannotator`.

        ``detailed_label_quality`` : pandas.DataFrame
            Similar to output as `~cleanlab.multiannotator.get_label_quality_multiannotator`.

        ``annotator_stats`` : pandas.DataFrame
            Similar to output as `~cleanlab.multiannotator.get_label_quality_multiannotator`.

        ``model_weight`` : np.ndarray
            Only returned if `return_weights=True`.
            An array of shape ``(P,)`` where is the number of models in the ensemble, specifying the weight of each classifier model in weighted averages used to estimate label quality.
            These weigthts is an estimate of how trustworthy the model is relative the annotators.
            An array of shape ``(P,)`` where is the number of models in the ensemble, specifying the model weight used in weighted averages.

        ``annotator_weight`` : np.ndarray
            Only returned if `return_weights=True`.
            Similar to output as `~cleanlab.multiannotator.get_label_quality_multiannotator`.

    See Also
    --------
    get_label_quality_multiannotator
    """
    if isinstance(labels_multiannotator, pd.DataFrame):
        annotator_ids = labels_multiannotator.columns
        index_col = labels_multiannotator.index
        labels_multiannotator = (
            labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
        )
    elif isinstance(labels_multiannotator, np.ndarray):
        annotator_ids = None
        index_col = None
    else:
        raise ValueError("labels_multiannotator must be either a NumPy array or Pandas DataFrame.")

    
    num_annotations = np.sum(~np.isnan(labels_multiannotator), axis=1)

    # temp scale pred_probs
    if calibrate_probs:
        for i in range(len(pred_probs)):
            curr_pred_probs = pred_probs[i]
            optimal_temp = find_best_temp_scaler(labels_multiannotator, curr_pred_probs)
            pred_probs[i] = temp_scale_pred_probs(curr_pred_probs, optimal_temp)

    label_quality = pd.DataFrame({"num_annotations": num_annotations}, index=index_col)

    avg_pred_probs = np.mean(pred_probs, axis=0)
    majority_vote_label = get_majority_vote_label(
        labels_multiannotator=labels_multiannotator,
        pred_probs=avg_pred_probs,
        verbose=False,
    )
    (
        MV_annotator_agreement,
        MV_consensus_quality_score,
        MV_post_pred_probs,
        MV_model_weight,
        MV_annotator_weight,
    ) = _get_consensus_stats(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        num_annotations=num_annotations,
        consensus_label=majority_vote_label,
        verbose=verbose,
        ensemble=True,
        **label_quality_score_kwargs,
    )

    consensus_label = np.full(len(majority_vote_label), np.nan)
    for i in range(len(consensus_label)):
        max_pred_probs_ind = np.where(MV_post_pred_probs[i] == np.max(MV_post_pred_probs[i]))[0]
        if len(max_pred_probs_ind) == 1:
            consensus_label[i] = max_pred_probs_ind[0]
        else:
            consensus_label[i] = majority_vote_label[i]
    consensus_label = consensus_label.astype(int)  # convert all label types to int

    (
        annotator_agreement,
        consensus_quality_score,
        post_pred_probs,
        model_weight,
        annotator_weight,
    ) = _get_consensus_stats(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        num_annotations=num_annotations,
        consensus_label=consensus_label,
        verbose=verbose,
        ensemble=True,
        **label_quality_score_kwargs,
    )

    if verbose:
        check_consensus_label_classes(
            labels_multiannotator=labels_multiannotator,
            consensus_label=consensus_label,
            consensus_method="crowdlab",
        )

    (
        label_quality["consensus_label"],
        label_quality["consensus_quality_score"],
        label_quality["annotator_agreement"],
    ) = (
        consensus_label,
        consensus_quality_score,
        annotator_agreement,
    )

    label_quality = label_quality.reindex(
        columns=[
            "consensus_label",
            "consensus_quality_score",
            "annotator_agreement",
            "num_annotations",
        ]
    )

    detailed_label_quality = None

    if return_detailed_quality:
        detailed_label_quality = np.apply_along_axis(
            _get_annotator_label_quality_score,
            axis=0,
            arr=labels_multiannotator,
            pred_probs=post_pred_probs,
            label_quality_score_kwargs=label_quality_score_kwargs,
        )
        detailed_label_quality_df = pd.DataFrame(
            detailed_label_quality, index=index_col, columns=annotator_ids
        ).add_prefix("quality_annotator_")

    if return_annotator_stats:
        annotator_stats = _get_annotator_stats(
            labels_multiannotator=labels_multiannotator,
            pred_probs=post_pred_probs,
            consensus_label=consensus_label,
            num_annotations=num_annotations,
            annotator_agreement=annotator_agreement,
            model_weight=np.mean(model_weight),  
            annotator_weight=annotator_weight,
            consensus_quality_score=consensus_quality_score,
            detailed_label_quality=detailed_label_quality,
            annotator_ids=annotator_ids,
        )

    labels_info = {
        "label_quality": label_quality,
    }

    if return_detailed_quality:
        labels_info["detailed_label_quality"] = detailed_label_quality_df
    if return_annotator_stats:
        labels_info["annotator_stats"] = annotator_stats
    if return_weights:
        labels_info["model_weight"] = model_weight
        labels_info["annotator_weight"] = annotator_weight

    return labels_info


def get_active_learning_scores(
    labels_multiannotator: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    pred_probs: Optional[np.ndarray] = None,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an ActiveLab quality score for each example in the dataset, to estimate which examples are most informative to (re)label next in active learning.

    We consider settings where one example can be labeled by one or more annotators and some examples have no labels at all so far.

    The score is in between 0 and 1, and can be used to prioritize what data to collect additional labels for.
    Lower scores indicate examples whose true label we are least confident about based on the current data;
    collecting additional labels for these low-scoring examples will be more informative than collecting labels for other examples.
    To use an annotation budget most efficiently, select a batch of examples with the lowest scores and collect one additional label for each example,
    and repeat this process after retraining your classifier.

    You can use this function to get active learning scores for: examples that already have one or more labels (specify ``labels_multiannotator`` and ``pred_probs``
    as arguments), or for unlabeled examples (specify ``pred_probs_unlabeled``), or for both types of examples (specify all of the above arguments).

    To analyze a fixed dataset labeled by multiple annotators rather than collecting additional labels, try the
    `~cleanlab.multiannotator.get_label_quality_multiannotator` (CROWDLAB) function instead.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray, optional
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators. Note that this function also works with
        datasets where there is only one annotator (M=1).
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
        Note that examples that have no annotator labels should not be included in this DataFrame/array.
        This argument is optional if ``pred_probs`` is not provided (you might only provide ``pred_probs_unlabeled`` to only get active learning scores for the unlabeled examples).
    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
        This argument is optional if you only want to get active learning scores for unlabeled examples (specify only ``pred_probs_unlabeled`` instead).
    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model for examples that have no annotator labels.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
        This argument is optional if you only want to get active learning scores for already-labeled examples (specify only ``pred_probs`` instead).

    Returns
    -------
    active_learning_scores : np.ndarray
        Array of shape ``(N,)`` indicating the ActiveLab quality scores for each example.
        This array is empty if no already-labeled data was provided via ``labels_multiannotator``.
        Examples with the lowest scores are those we should label next in order to maximally improve our classifier model.

    active_learning_scores_unlabeled : np.ndarray
        Array of shape ``(N,)`` indicating the active learning quality scores for each unlabeled example.
        Returns an empty array if no unlabeled data is provided.
        Examples with the lowest scores are those we should label next in order to maximally improve our classifier model
        (scores for unlabeled data are directly comparable with the `active_learning_scores` for labeled data).
    """
    
    import numpy as np
    
    assert_valid_pred_probs(pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled)

    if pred_probs is not None:
        if labels_multiannotator is None:
            raise ValueError(
                "labels_multiannotator cannot be None when passing in pred_probs. ",
                "Either provide labels_multiannotator to obtain active learning scores for the labeled examples, "
                "or just pass in pred_probs_unlabeled to get active learning scores for unlabeled examples.",
            )

        if isinstance(labels_multiannotator, pd.DataFrame):
            labels_multiannotator = (
                labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
            )
        elif not isinstance(labels_multiannotator, np.ndarray):
            raise ValueError(
                "labels_multiannotator must be either a NumPy array or Pandas DataFrame."
            )


        num_classes = get_num_classes(pred_probs=pred_probs)
        
        if (np.sum(~np.isnan(labels_multiannotator), axis=1) == 1).all():
            optimal_temp = 1.0  


            consensus_label = get_majority_vote_label(
                labels_multiannotator=labels_multiannotator,
                pred_probs=pred_probs,
                verbose=False,
            )
            quality_of_consensus_labeled = get_label_quality_scores(consensus_label, pred_probs)
            model_weight = 1
            annotator_weight = np.full(labels_multiannotator.shape[1], 1)
            avg_annotator_weight = np.mean(annotator_weight)

        # examples are annotated by multiple annotators
        else:
            optimal_temp = find_best_temp_scaler(labels_multiannotator, pred_probs)
            pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)

            multiannotator_info = get_label_quality_multiannotator(
                labels_multiannotator,
                pred_probs,
                return_annotator_stats=False,
                return_detailed_quality=False,
                return_weights=True,
            )

            quality_of_consensus_labeled = multiannotator_info["label_quality"][
                "consensus_quality_score"
            ]
            model_weight = multiannotator_info["model_weight"]
            #print(model_weight)
            annotator_weight = multiannotator_info["annotator_weight"]
            #print('aw',type(annotator_weight))
            avg_annotator_weight = np.mean(annotator_weight)
            print(type(quality_of_consensus_labeled), quality_of_consensus_labeled)


            #print(quality_of_consensus_labeled.ndim)
            #print(avg_annotator_weight)

            N, M, K = labels_multiannotator.shape  
            #active_learning_scores = np.zeros(N)
            avg_annotator_weight = np.mean(annotator_weight)
            import numpy as np

# Sample shapes for demonstration
            N, M, K = labels_multiannotator.shape  # Example values for the number of examples, annotators, and classes
            #num_classes = K  # This variable seems to be used in your original calculation

            # Initialize the output structure
            active_learning_scores = np.zeros((N, K))

            # Sample data initialization (replace with your actual data)
            labels_multiannotator = np.random.randint(0, 2, (N, M, K))  # Example binary labels
            quality_of_consensus_labeled = np.random.rand(N)  # Example quality scores
            annotator_weight = np.random.rand(M, K)  # Example annotator weights per class
            model_weight = 1.0  # Example model weight
            avg_annotator_weight = np.mean(annotator_weight, axis=0)  # Average weight per class

            # Calculate active learning scores for each class within each example
            for i in range(N):  # For each example
                for j in range(K):  # For each class
                    # Extract annotator labels for this class across all annotators
                    annotator_labels_for_class = labels_multiannotator[i, :, j]
                    
                    # Calculate the sum of weights for annotators that provided a label (1) for this class
                    sum_annotator_weights = np.sum(annotator_weight[:, j][annotator_labels_for_class == 1])
                    
                    # Calculate the weighted average using the consensus quality and a baseline difficulty (1 / num_classes)
                    # Adjust the calculation as necessary to fit the multi-label context
                    active_learning_scores[i, j] = np.average(
                        [quality_of_consensus_labeled[i], 1 / num_classes],
                        weights=[
                            sum_annotator_weights + model_weight,
                            avg_annotator_weight[j]  # Use the average weight for this specific class
                        ]
                    )
        
# Now, active_learning_scores is an array of shape (N, K) where each element [i, j] is the active learning score
# for the i-th example and j-th class

            '''# Initialize the active learning scores array
            active_learning_scores = np.full((N, K), np.nan)
            active_learning=[]
            for i in range(N):
    # Loop through each class within the example
                for j in range(K):
                    # Extract the labels for the current class from all annotators
                    annotator_labels_for_class = labels_multiannotator[i, :, j]
                    
                    # Calculate the sum of weights for annotators that labeled the current class
                    # Note: This assumes annotator_weight indicates the reliability of an annotator's label for each class
                    # And that labels are not NaN, since we're dealing with binary/multilabel data, this might need adjusting
                    column_label = annotator_weight.columns[j]  # Get column label by position
                    weights_for_class = np.sum(annotator_weight[column_label][annotator_labels_for_class == 1])

                    #weights_for_class = np.sum(annotator_weight[:, j][annotator_labels_for_class == 1])
                    
                    # If no annotator labeled the current class, use a default weight (e.g., 0)
                    if weights_for_class == 0:
                        weights_for_class = np.nan  # Adjust based on how you want to handle no labels
                    
                    # Calculate the active learning score for the current class
                    # Note: This calculation will need to be adjusted if handling NaN values differently
                    active_learning[i].append( np.average(
                        [quality_of_consensus_labeled[i], 1 / num_classes],
                        weights=[
                            weights_for_class + model_weight,
                            avg_annotator_weight
                        ]
                    ))
            active_learning_scores=np.array(active_learning)
        #return active_learning_scores,active_learning_scores_unlabeled'''
            
            
                





    elif pred_probs_unlabeled is not None:
        num_classes = get_num_classes(pred_probs=pred_probs_unlabeled)
        optimal_temp = 1
        model_weight = 1
        avg_annotator_weight = 1
        active_learning_scores = np.array([])

    else:
        raise ValueError(
            "pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two."
        )

    if pred_probs_unlabeled is not None:
        pred_probs_unlabeled = temp_scale_pred_probs(pred_probs_unlabeled, optimal_temp)
        quality_of_consensus_unlabeled = np.max(pred_probs_unlabeled, axis=1)

        active_learning_scores_unlabeled = np.average(
            np.stack(
                [
                    quality_of_consensus_unlabeled,
                    np.full(len(quality_of_consensus_unlabeled), 1 / num_classes),
                ]
            ),
            weights=[model_weight, avg_annotator_weight],
            axis=0,
        )

    else:
        active_learning_scores_unlabeled = np.array([])

    return active_learning_scores, active_learning_scores_unlabeled



def break_tie(label_mode, pred_probs, threshold=0.5):
    max_pred_probs = np.where(pred_probs[label_mode] >= threshold)[0]
    
    if len(max_pred_probs) > 0:
        return label_mode[max_pred_probs[0]]
    else:
        return 0

def get_majority_vote_label(
    labels_multiannotator,
    pred_probs,
    verbose: bool = True 
) -> np.ndarray:

    a = []
    num_classes = pred_probs.shape[1]
    for i in range(labels_multiannotator.shape[0]):
        most_repeating_values = []

        for column in labels_multiannotator[i].T:

            non_nan_values = column[~np.isnan(column)].astype(int)
            counts = np.bincount(non_nan_values, minlength=num_classes)
            max_count = np.max(counts)
            indices = np.where(counts == max_count)[0]


            if len(indices) > 1:
                tied_label = break_tie(indices, pred_probs[i])
                most_repeating_values.append(tied_label)
            else:
                most_repeating_values.append(indices[0])

        a.append(most_repeating_values)

    return np.array(a)


def convert_long_to_wide_dataset(
    labels_multiannotator_long: pd.DataFrame,
) -> pd.DataFrame:
    """Converts a long format dataset to wide format which is suitable for passing into
    `~cleanlab.multiannotator.get_label_quality_multiannotator`.

    Dataframe must contain three columns named:

    #. ``task`` representing each example labeled by the annotators
    #. ``annotator`` representing each annotator
    #. ``label`` representing the label given by an annotator for the corresponding task (i.e. example)

    Parameters
    ----------
    labels_multiannotator_long : pd.DataFrame
        pandas DataFrame in long format with three columns named ``task``, ``annotator`` and ``label``

    Returns
    -------
    labels_multiannotator_wide : pd.DataFrame
        pandas DataFrame of the proper format to be passed as ``labels_multiannotator`` for the other ``cleanlab.multiannotator`` functions.
    """
    labels_multiannotator_wide = labels_multiannotator_long.pivot(
        index="task", columns="annotator", values="label"
    )
    labels_multiannotator_wide.index.name = None
    labels_multiannotator_wide.columns.name = None
    return labels_multiannotator_wide


def _get_consensus_stats(
    labels_multiannotator: np.ndarray,
    pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    consensus_label: np.ndarray,
    quality_method: str = "crowdlab",
    verbose: bool = True,
    ensemble: bool = False,
    label_quality_score_kwargs: dict = {},
) -> tuple:
    """Returns a tuple containing the consensus labels, annotator agreement scores, and quality of consensus

    Parameters
    ----------
    labels_multiannotator : np.ndarray
        2D numpy array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view `~cleanlab.multiannotator.get_label_quality_multiannotator`
    label_quality_score_kwargs : dict, optional
        Keyword arguments to pass into ``get_label_quality_scores()``.
    verbose : bool, default = True
        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.
    ensemble : bool, default = False
        Boolean flag to indicate whether the pred_probs passed are from ensemble models.

    Returns
    ------
    stats : tuple
        A tuple of (consensus_label, annotator_agreement, consensus_quality_score, post_pred_probs).
    """

    # compute the fraction of annotator agreeing with the consensus labels
    annotator_agreement = _get_annotator_agreement_with_consensus(
        labels_multiannotator=labels_multiannotator,
        consensus_label=consensus_label,
    )
    if 0:
        pass
    else:
        post_pred_probs, model_weight, annotator_weight = _get_post_pred_probs_and_weights(
            labels_multiannotator=labels_multiannotator,
            consensus_label=consensus_label,
            prior_pred_probs=pred_probs,
            num_annotations=num_annotations,
            annotator_agreement=annotator_agreement,
            quality_method=quality_method,
            verbose=verbose,
        )



    # compute quality of the consensus labels
    consensus_quality_score = _get_consensus_quality_score(
        consensus_label=consensus_label,
        pred_probs=post_pred_probs,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        quality_method=quality_method,
        label_quality_score_kwargs=label_quality_score_kwargs,
    )

    return (
        annotator_agreement,
        consensus_quality_score,
        post_pred_probs,
        model_weight,
        annotator_weight,
    )


def _get_annotator_stats(
    labels_multiannotator: np.ndarray,
    pred_probs: np.ndarray,
    consensus_label: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    model_weight: np.ndarray,
    annotator_weight: np.ndarray,
    consensus_quality_score: np.ndarray,
    detailed_label_quality: Optional[np.ndarray] = None,
    annotator_ids: Optional[pd.Index] = None,
    quality_method: str = "crowdlab",
) -> pd.DataFrame:
    """Returns a dictionary containing overall statistics about each annotator.

    Parameters
    ----------
    labels_multiannotator : np.ndarray
        2D numpy array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    model_weight : float
        float specifying the model weight used in weighted averages,
        None if model weight is not used to compute quality scores
    annotator_weight : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,
        None if annotator weights are not used to compute quality scores
    consensus_quality_score : np.ndarray
        An array of shape ``(N,)`` with the quality score of the consensus.
    detailed_label_quality :
        pandas DataFrame containing the detailed label quality scores for all examples and annotators
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view `~cleanlab.multiannotator.get_label_quality_multiannotator`

    Returns
    -------
    annotator_stats : pd.DataFrame
        Overall statistics about each annotator.
        For details, see the documentation of `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    """

    annotator_quality = _get_annotator_quality(
        labels_multiannotator=labels_multiannotator,
        pred_probs=pred_probs,
        consensus_label=consensus_label,
        num_annotations=num_annotations,
        annotator_agreement=annotator_agreement,
        model_weight=model_weight,
        annotator_weight=annotator_weight,
        detailed_label_quality=detailed_label_quality,
        quality_method=quality_method,
    )

    # Compute the number of labels labeled/ by each annotator
    num_examples_labeled = np.sum(~np.isnan(labels_multiannotator), axis=0)
    num_examples_labeled=num_examples_labeled[:, 0]
    # Compute the fraction of labels annotated by each annotator that agrees with the consensus label
    
    agreement_with_consensus = np.zeros(labels_multiannotator.shape[1])
    for i in range(len(agreement_with_consensus)):
        labels = labels_multiannotator[:, i]
        labels_mask = ~np.isnan(labels)
        agreement_with_consensus[i] = np.mean(labels[labels_mask] == consensus_label[labels_mask])

    # Find the worst labeled class for each annotator
    worst_class = _get_annotator_worst_class(
        labels_multiannotator=labels_multiannotator,
        consensus_label=consensus_label,
        consensus_quality_score=consensus_quality_score,
    )
    annotators=['wkratochvil', 'kmaisuria', 'hannahweisman', 'ezequielbautista','aribahali', 'kaelynnrodriguez', 'babatundeshofolu','rishika.patel@ufl.edu\\health', 'jennifer.noa']


    # Create multi-annotator stats DataFrame from its columns
    annotator_stats = pd.DataFrame(
        {
            "annotator_quality": annotator_quality,
            "agreement_with_consensus": agreement_with_consensus,
            "worst_class": worst_class.tolist(),
            "num_of_exampels_labelled":num_examples_labeled
            
        },
        index=annotators,
    )
    
    return annotator_stats.sort_values(by=["annotator_quality", "agreement_with_consensus"])


def _get_annotator_agreement_with_consensus(
    labels_multiannotator: np.ndarray,
    consensus_label: np.ndarray,
) -> np.ndarray:
    """Returns the fractions of annotators that agree with the consensus label per example. Note that the
    fraction for each example only considers the annotators that labeled that particular example.

    Parameters
    ----------
    labels_multiannotator : np.ndarray
        2D numpy array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.

    Returns
    -------
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    """

    num_examples, num_annotators, num_classes = labels_multiannotator.shape

    annotator_agreement = np.zeros((num_examples, num_classes))

    for i in range(num_examples):
        for j in range(num_classes):
            labels_for_class = labels_multiannotator[i, :, j]

            valid_labels = labels_for_class[~np.isnan(labels_for_class)]

            if len(valid_labels) > 0:
                agreement_fraction = np.mean(valid_labels == consensus_label[i, j])
                annotator_agreement[i, j] = agreement_fraction

    return annotator_agreement




def _get_single_annotator_agreement(labels_multiannotator, num_annotations, annotator_idx):
    K = labels_multiannotator.shape[2]
    annotator_agreement_per_label = np.zeros(K)
    
    for k in range(K):
        annotator_agreement_per_example = np.zeros(len(labels_multiannotator))
        for i, labels in enumerate(labels_multiannotator):
            labels_subset = labels[~np.isnan(labels[:, k])]
            examples_num_annotators = len(labels_subset)
            if examples_num_annotators > 1:
                match_counts = np.sum(labels_subset[:, k] == labels[annotator_idx, k])
                annotator_agreement_per_example[i] = (match_counts - 1) / (examples_num_annotators - 1)
        adjusted_num_annotations = num_annotations - 1
        annotator_agreement = np.nan if np.sum(adjusted_num_annotations) == 0 else np.average(annotator_agreement_per_example, weights=adjusted_num_annotations)
        annotator_agreement_per_label[k] = annotator_agreement

    return annotator_agreement_per_label

def _get_annotator_agreement_with_annotators(
    labels_multiannotator: np.ndarray,
    num_annotations: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    M, K = labels_multiannotator.shape[1], labels_multiannotator.shape[2]
    annotator_agreement_with_annotators = np.zeros((M, K))

    for i in range(M):
        annotator_labels = labels_multiannotator[:, i, :]
        annotator_labels_mask = ~np.isnan(annotator_labels).all(axis=1)
        filtered_labels = labels_multiannotator[annotator_labels_mask, :, :]
        filtered_num_annotations = num_annotations[annotator_labels_mask]
        annotator_agreement_with_annotators[i] = _get_single_annotator_agreement(filtered_labels, filtered_num_annotations, i)

    return annotator_agreement_with_annotators


def _get_post_pred_probs_and_weights(
    labels_multiannotator: np.ndarray,
    consensus_label: np.ndarray,
    prior_pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_method: str = "crowdlab",
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[float], Optional[np.ndarray]]:
    """Return the posterior predicted probabilities of each example given a specified quality method.

    Parameters
    ----------
    labels_multiannotator : np.ndarray
        2D numpy array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    prior_pred_probs : np.ndarray
        An array of shape ``(N, K)`` of prior predicted probabilities, ``P(label=k|x)``, usually the out-of-sample predicted probability computed by a model.
        For details, predicted probabilities in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view `~cleanlab.multiannotator.get_label_quality_multiannotator`
    verbose : bool, default = True
        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.

    Returns
    -------
    post_pred_probs : np.ndarray
        An array of shape ``(N, K)`` with the posterior predicted probabilities.

    model_weight : float
        float specifying the model weight used in weighted averages,
        None if model weight is not used to compute quality scores

    annotator_weight : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,
        None if annotator weights are not used to compute quality scores

    """
    valid_methods = [
        "crowdlab",
        "agreement",
    ]

    # setting dummy variables for model and annotator weights that will be returned
    # only relevant for quality_method == crowdlab, return None for all other methods
    return_model_weight = None
    return_annotator_weight = None

    if quality_method == "crowdlab":
        num_classes = get_num_classes(pred_probs=prior_pred_probs)

        # likelihood that any annotator will or will not annotate the consensus label for any example
        consensus_likelihood = np.mean(annotator_agreement[num_annotations != 1])
        non_consensus_likelihood = (1 - consensus_likelihood) / (num_classes - 1)
  
        mask = num_annotations != 1
  
        consensus_label_subset = consensus_label[mask]
        prior_pred_probs_subset = prior_pred_probs[mask]
        most_likely_class_error=[]
        num_positions = len(consensus_label_subset[0])
        

        for position in range(num_positions):
            labels_at_position = [labels[position] for labels in consensus_label_subset]
            
            most_common_label = np.argmax(np.bincount(labels_at_position, minlength=num_classes))

            error_rate = np.mean([label != most_common_label for label in labels_at_position])
            most_likely_class_error.append(error_rate)


    
        most_likely_class_error = np.array(most_likely_class_error)
        most_likely_class_error = np.clip(most_likely_class_error, a_min=CLIPPING_LOWER_BOUND, a_max=None)

        annotator_agreement_with_annotators = _get_annotator_agreement_with_annotators(
            labels_multiannotator, num_annotations,verbose
        )

        annotator_error = 1 - annotator_agreement_with_annotators
        adjusted_annotator_agreement = np.clip(
            1 - (annotator_error / most_likely_class_error), a_min=CLIPPING_LOWER_BOUND, a_max=None
        )
        
        prob_mask = prior_pred_probs_subset > 0.5
    
   
        adjusted_labels = np.where(prob_mask, consensus_label_subset, 1 - consensus_label_subset)

        model_error=np.mean(adjusted_labels, axis=0)
        relative_performance = 1 - (model_error / most_likely_class_error)
        clipped_performance = np.maximum(relative_performance, CLIPPING_LOWER_BOUND)
        sqrt_mean_annotations = np.sqrt(np.mean(num_annotations))
        model_weight = clipped_performance * sqrt_mean_annotations


        post_pred_probs=prior_pred_probs
        return_model_weight = model_weight
        return_annotator_weight = adjusted_annotator_agreement

    elif quality_method == "agreement":
        num_classes = get_num_classes(pred_probs=prior_pred_probs)
        label_counts = np.full((len(labels_multiannotator), num_classes), np.NaN)
        for i, labels in enumerate(labels_multiannotator):
            label_counts[i, :] = value_counts(labels[~np.isnan(labels)], num_classes=num_classes)

        post_pred_probs = label_counts / num_annotations.reshape(-1, 1)

    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid quality method!
            Please choose a valid quality_method: {valid_methods}
            """
        )

    return post_pred_probs, return_model_weight, return_annotator_weight


def _get_consensus_quality_score(
    consensus_label: np.ndarray,
    pred_probs: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    quality_method: str = "crowdlab",
    label_quality_score_kwargs: dict = {},
) -> np.ndarray:
    #print('cl',consensus_label)
    """Return scores representing quality of the consensus label for each example.

    Parameters
    ----------
    labels_multiannotator : np.ndarray
        2D numpy array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of posterior predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the consensus label.
        For valid quality methods, view `~cleanlab.multiannotator.get_label_quality_multiannotator`

    Returns
    -------
    consensus_quality_score : np.ndarray
        An array of shape ``(N,)`` with the quality score of the consensus.
    """

    valid_methods = [
        "crowdlab",
        "agreement",
    ]

    if quality_method == "crowdlab":
        consensus_quality_score = get_label_quality_scores(
            consensus_label, pred_probs, **label_quality_score_kwargs
        )

    elif quality_method == "agreement":
        consensus_quality_score = annotator_agreement

    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid consensus quality method!
            Please choose a valid quality_method: {valid_methods}
            """
        )

    return consensus_quality_score


def _get_annotator_label_quality_score(
    annotator_label: np.ndarray,
    pred_probs: np.ndarray,
    label_quality_score_kwargs: dict = {},
) -> np.ndarray:
    """Returns quality scores for each datapoint.
    Very similar functionality as ``_get_consensus_quality_score`` with additional support for annotator labels that contain NaN values.
    For more info about parameters and returns, see the docstring of `~cleanlab.multiannotator._get_consensus_quality_score`.
    """
    mask = ~np.isnan(annotator_label).any(axis=1)

    filtered_labels = annotator_label[mask].astype(int)
    
    filtered_pred_probs = pred_probs[mask]
    
    annotator_label_quality_score_subset = get_label_quality_scores(
        labels=filtered_labels,
        pred_probs=filtered_pred_probs,
        **label_quality_score_kwargs
    )

    annotator_label_quality_score = np.full(annotator_label.shape[0], np.nan)
    annotator_label_quality_score[mask] = annotator_label_quality_score_subset

    return annotator_label_quality_score




def _get_annotator_quality(
    labels_multiannotator: np.ndarray,
    pred_probs: np.ndarray,
    consensus_label: np.ndarray,
    num_annotations: np.ndarray,
    annotator_agreement: np.ndarray,
    model_weight: np.ndarray,
    annotator_weight: np.ndarray,
    detailed_label_quality: Optional[np.ndarray] = None,
    quality_method: str = "crowdlab",
) -> pd.DataFrame:
    """Returns annotator quality score for each annotator.

    Parameters
    ----------
    labels_multiannotator : np.ndarray
        2D numpy array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    num_annotations : np.ndarray
        An array of shape ``(N,)`` with the number of annotators that have labeled each example.
    annotator_agreement : np.ndarray
        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.
    model_weight : float
        An array of shape ``(P,)`` where P is the number of models in this ensemble, specifying the model weight used in weighted averages,
        ``None`` if model weight is not used to compute quality scores
    annotator_weight : np.ndarray
        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,
        ``None`` if annotator weights are not used to compute quality scores
    detailed_label_quality :
        pandas DataFrame containing the detailed label quality scores for all examples and annotators
    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])
        Specifies the method used to calculate the quality of the annotators.
        For valid quality methods, view `~cleanlab.multiannotator.get_label_quality_multiannotator`

    Returns
    -------
    annotator_quality : np.ndarray
        Quality scores of a given annotator's labels
    """

    valid_methods = [
        "crowdlab",
        "agreement",
    ]

    if quality_method == "crowdlab":
        if detailed_label_quality is None:
            annotator_lqs = np.zeros(labels_multiannotator.shape[1])
            for i in range(len(annotator_lqs)):
                labels = labels_multiannotator[:, i]
                labels_mask = ~np.isnan(labels)
                annotator_lqs[i] = np.mean(
                    get_label_quality_scores(
                        labels[labels_mask].astype(int),
                        pred_probs[labels_mask],
                    )
                )
        else:
            annotator_lqs = np.nanmean(detailed_label_quality, axis=0)

        mask = num_annotations != 1
        labels_multiannotator_subset = labels_multiannotator[mask]
        consensus_label_subset = consensus_label[mask]

        annotator_agreement = np.zeros(labels_multiannotator_subset.shape[1])
        for i in range(len(annotator_agreement)):
            labels = labels_multiannotator_subset[:, i]
            labels_mask = ~np.isnan(labels)
            # case where annotator does not annotate any examples with any other annotators
            # TODO: do we want to impute the mean or just return np.nan
            if np.sum(labels_mask) == 0:
                annotator_agreement[i] = np.NaN
            else:
                annotator_agreement[i] = np.mean(
                    labels[labels_mask] == consensus_label_subset[labels_mask],
                )
        #print('aa',annotator_agreement)
        #avg_num_annotations_frac=[]
        avg_num_annotations_frac = np.mean(num_annotations) / len(annotator_weight)
        annotator_weight_adjusted = np.sum(annotator_weight) * avg_num_annotations_frac

        w = model_weight / (model_weight + annotator_weight_adjusted)
        w=w[0]
        
        annotator_quality = w * annotator_lqs + (1 - w) * annotator_agreement

    elif quality_method == "agreement":
        mask = num_annotations != 1
        labels_multiannotator_subset = labels_multiannotator[mask]
        consensus_label_subset = consensus_label[mask]

        annotator_quality = np.zeros(labels_multiannotator_subset.shape[1])
        for i in range(len(annotator_quality)):
            labels = labels_multiannotator_subset[:, i]
            labels_mask = ~np.isnan(labels)
            # case where annotator does not annotate any examples with any other annotators
            if np.sum(labels_mask) == 0:
                annotator_quality[i] = np.NaN
            else:
                annotator_quality[i] = np.mean(
                    labels[labels_mask] == consensus_label_subset[labels_mask],
                )

    else:
        raise ValueError(
            f"""
            {quality_method} is not a valid annotator quality method!
            Please choose a valid quality_method: {valid_methods}
            """
        )
    #print('aaaa',annotator_quality)
    return annotator_quality


def _get_annotator_worst_class(
    labels_multiannotator: np.ndarray,
    consensus_label: np.ndarray,
    consensus_quality_score: np.ndarray,
) -> np.ndarray:
    """Returns the class which each annotator makes the most errors in.

    Parameters
    ----------
    labels_multiannotator : np.ndarray
        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    consensus_quality_score : np.ndarray
        An array of shape ``(N,)`` with the quality score of the consensus.

    Returns
    -------
    worst_class : np.ndarray
        The class that is most frequently mislabeled by a given annotator.
    """

    worst_class = np.apply_along_axis(
        _get_single_annotator_worst_class,
        axis=0,
        arr=labels_multiannotator,
        consensus_label=consensus_label,
        consensus_quality_score=consensus_quality_score,
    ).astype(int)

    return worst_class


def _get_single_annotator_worst_class(
    labels: np.ndarray,
    consensus_label: np.ndarray,
    consensus_quality_score: np.ndarray,
) -> int:
    """Returns the class a given annotator makes the most errors in.

    Parameters
    ----------
    labels : np.ndarray
        An array of shape ``(N,)`` with the labels from the annotator we want to evaluate.
    consensus_label : np.ndarray
        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.
    consensus_quality_score : np.ndarray
        An array of shape ``(N,)`` with the quality score of the consensus.

    Returns
    -------
    worst_class : int
        The class that is most frequently mislabeled by the given annotator.
    """
    labels = pd.Series(labels)
    labels_mask = pd.notna(labels)
    class_accuracies = (labels[labels_mask] == consensus_label[labels_mask]).groupby(labels).mean()
    accuracy_min_idx = class_accuracies[class_accuracies == class_accuracies.min()].index.values

    if len(accuracy_min_idx) == 1:
        return accuracy_min_idx[0]

    # tiebreak 1: class counts
    class_count = labels[labels_mask].groupby(labels).count()[accuracy_min_idx]
    count_max_idx = class_count[class_count == class_count.max()].index.values

    if len(count_max_idx) == 1:
        return count_max_idx[0]

    # tiebreak 2: consensus quality scores
    avg_consensus_quality = (
        pd.DataFrame(
            {"annotator_label": labels, "consensus_quality_score": consensus_quality_score}
        )[labels_mask]
        .groupby("annotator_label")
        .mean()["consensus_quality_score"][count_max_idx]
    )
    quality_max_idx = avg_consensus_quality[
        avg_consensus_quality == avg_consensus_quality.max()
    ].index.values

    # return first item even if there are ties - no better methods to tiebreak
    return quality_max_idx[0]