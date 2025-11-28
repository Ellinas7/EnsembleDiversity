import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# =============================================================================
# IMPORT DATASET
# =============================================================================
from datasets.dataset import dataset

# =============================================================================
# IMPORT ALGORITMI ML
# =============================================================================
from ML_algorithms.decision_tree import decision_tree
from ML_algorithms.random_forest import random_forest
from ML_algorithms.xgboost import xgboost
from ML_algorithms.adaboost import adaboost
from ML_algorithms.extra_trees import extra_trees
from ML_algorithms.gradient_boosting_decision_trees import gradient_boosting_decision_trees
from ML_algorithms.light_gbm import light_gbm
from ML_algorithms.catboost import catboost
from ML_algorithms.rotation_forest import rotation_forest
from ML_algorithms.random_rotation_forest import random_rotation_forest
from ML_algorithms.ML_algorithm import ML_algorithm

# =============================================================================
# IMPORT DECORATORI REJECTION
# =============================================================================
from ML_algorithms.static_threshold_rejection_decorator import static_threshold_rejection_decorator
from ML_algorithms.percentile_threshold_rejection_decorator import percentile_threshold_rejection_decorator

# =============================================================================
# IMPORT METRICHE - Singolo Classificatore
# =============================================================================
from diversity_metrics.single_correct_prediction_rate import single_correct_prediction_rate
from diversity_metrics.single_misclassification_rate import single_misclassification_rate
from diversity_metrics.single_rejection_rate import single_rejection_rate
from diversity_metrics.hit import hit
from diversity_metrics.miss import miss
from diversity_metrics.acceptance_accuracy import acceptance_accuracy
from diversity_metrics.performance_loss import performance_loss
from diversity_metrics.diversity_metric import diversity_metric

# =============================================================================
# IMPORT METRICHE - Coppia Classificatori (Voting 2/2)
# =============================================================================
from diversity_metrics.double_correct_prediction_rate import double_correct_prediction_rate
from diversity_metrics.double_wrong_prediction_rate import double_wrong_prediction_rate
from diversity_metrics.double_rejection_rate import double_rejection_rate

# =============================================================================
# IMPORT METRICHE - Coppia Classificatori (Recovery Block)
# =============================================================================
from diversity_metrics.recovery_rate import recovery_rate
from diversity_metrics.recovery_failure_rate import recovery_failure_rate
from diversity_metrics.recovery_rejection_rate import recovery_rejection_rate
from diversity_metrics.recovery_correct_prediction_rate import recovery_correct_prediction_rate
from diversity_metrics.recovery_wrong_prediction_rate import recovery_wrong_prediction_rate
from diversity_metrics.recovery_rejection_prediction_rate import recovery_rejection_prediction_rate

# =============================================================================
# IMPORT METRICHE - Terna Classificatori (Majority Voting)
# =============================================================================
from diversity_metrics.majority_voting_correct_prediction_rate import majority_voting_correct_prediction_rate
from diversity_metrics.majority_voting_wrong_prediction_rate import majority_voting_wrong_prediction_rate
from diversity_metrics.majority_voting_rejection_prediction_rate import majority_voting_rejection_prediction_rate

# =============================================================================
# IMPORT METRICHE - Terna Classificatori (Voting 1/N)
# =============================================================================
from diversity_metrics.single_vote_correct_prediction_rate import single_vote_correct_prediction_rate
from diversity_metrics.single_vote_wrong_prediction_rate import single_vote_wrong_prediction_rate
from diversity_metrics.single_vote_rejection_rate import single_vote_rejection_rate
from diversity_metrics.ideal_single_vote_rate import ideal_single_vote_rate

# =============================================================================
# IMPORT METRICHE CLASSICHE DI DIVERSITY
# =============================================================================
from diversity_metrics.Q_statistic import Q_statistic
from diversity_metrics.disagreement_measure import disagreement_measure
from diversity_metrics.double_fault_measure import double_fault_measure
from diversity_metrics.entropy_measure import entropy_measure
from diversity_metrics.kohavi_wolpert_variance import kohavi_wolpert_variance
from diversity_metrics.generalized_diversity import generalized_diversity
from diversity_metrics.coincident_failure_diversity import coincident_failure_diversity


class metric_calculator:
    """Calcola metriche di diversity su un ensemble addestrato"""
    warnings.filterwarnings("ignore")
    
    def calculate(self, ds: dataset, algorithm: ML_algorithm, 
              metric: diversity_metric) -> float:
        
        X_train, X_test, y_train, y_test = ds.data
    
        algorithm.train(X_train, y_train)
    
        predictions = algorithm.get_estimator_predictions(X_test)
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    
        # Converti a stringa per confronti consistenti
        predictions = predictions.astype(str)
        y_test_array = y_test_array.astype(str)
    
        try:
            return metric._compute(predictions, y_test_array, X_test, algorithm)
        except TypeError:
            return metric._compute(predictions, y_test_array)
    


if __name__ == "__main__":

    # Dataset
    ds = dataset("/Users/matteopascuzzo/Desktop/Datasets/Error Detection/arancino_all_scikit.csv",
                    dataset_name="arancino_all_scikit", stratify=True)
    ds.preprocess()
    
    calc = metric_calculator()
    
    # Esperimento
    xgboost = xgboost(n_estimators=2)
    xgboost_with_statistic_rej = static_threshold_rejection_decorator(xgboost, confidence_threshold=0.9)
    result = calc.calculate(ds, xgboost_with_statistic_rej, Q_statistic())
    print(f"Q-statistic: {result}\n")