# metric_calculator.py

import numpy as np
from typing import Union, List, Dict, Type
import warnings

# Import dataset
from datasets.dataset import dataset

# Import Ensemble base e tipi specifici
from ensembles.ensemble import Ensemble
from ensembles.voting_2of2_ensemble import Voting2of2Ensemble
from ensembles.recovery_block_ensemble import RecoveryBlockEnsemble
from ensembles.majority_voting_ensemble import MajorityVotingEnsemble
from ensembles.voting_1ofn_ensemble import Voting1ofNEnsemble

# Import rejection decorator base
from rejection_techniques.abstract_rejection_decorator import abstract_rejection_decorator

# Import metriche classiche
from diversity_metrics.Q_statistic import Q_statistic
from diversity_metrics.disagreement_measure import disagreement_measure
from diversity_metrics.double_fault_measure import double_fault_measure
from diversity_metrics.entropy_measure import entropy_measure
from diversity_metrics.kohavi_wolpert_variance import kohavi_wolpert_variance
from diversity_metrics.generalized_diversity import generalized_diversity
from diversity_metrics.coincident_failure_diversity import coincident_failure_diversity

# Import metriche singolo classificatore
from diversity_metrics.single_correct_prediction_rate import single_correct_prediction_rate
from diversity_metrics.single_misclassification_rate import single_misclassification_rate
from diversity_metrics.single_rejection_rate import single_rejection_rate
from diversity_metrics.hit import hit
from diversity_metrics.miss import miss
from diversity_metrics.acceptance_accuracy import acceptance_accuracy
from diversity_metrics.performance_loss import performance_loss

# Import metriche voting 2/2
from diversity_metrics.double_correct_prediction_rate import double_correct_prediction_rate
from diversity_metrics.double_wrong_prediction_rate import double_wrong_prediction_rate
from diversity_metrics.double_rejection_rate import double_rejection_rate

# Import metriche recovery block
from diversity_metrics.recovery_rate import recovery_rate
from diversity_metrics.recovery_failure_rate import recovery_failure_rate
from diversity_metrics.recovery_rejection_rate import recovery_rejection_rate
from diversity_metrics.recovery_correct_prediction_rate import recovery_correct_prediction_rate
from diversity_metrics.recovery_wrong_prediction_rate import recovery_wrong_prediction_rate
from diversity_metrics.recovery_rejection_prediction_rate import recovery_rejection_prediction_rate

# Import metriche majority voting
from diversity_metrics.majority_voting_correct_prediction_rate import majority_voting_correct_prediction_rate
from diversity_metrics.majority_voting_wrong_prediction_rate import majority_voting_wrong_prediction_rate
from diversity_metrics.majority_voting_rejection_prediction_rate import majority_voting_rejection_prediction_rate

# Import metriche voting 1/n
from diversity_metrics.single_vote_correct_prediction_rate import single_vote_correct_prediction_rate
from diversity_metrics.single_vote_wrong_prediction_rate import single_vote_wrong_prediction_rate
from diversity_metrics.single_vote_rejection_rate import single_vote_rejection_rate
from diversity_metrics.ideal_single_vote_rate import ideal_single_vote_rate


class metric_calculator:
    """Calcola metriche su ensemble o singoli classificatori con rejection"""
    
    warnings.filterwarnings("ignore")
    
    # Metriche classiche (sempre calcolate per ensemble)
    CLASSIC_METRICS = {
        'q_statistic': Q_statistic,
        'disagreement_measure': disagreement_measure,
        'double_fault_measure': double_fault_measure,
        'entropy_measure': entropy_measure,
        'kohavi_wolpert_variance': kohavi_wolpert_variance,
        'generalized_diversity': generalized_diversity,
        'coincident_failure_diversity': coincident_failure_diversity
    }
    
    # Metriche per singolo classificatore
    SINGLE_METRICS = {
        'single_correct_prediction_rate': single_correct_prediction_rate,
        'single_misclassification_rate': single_misclassification_rate,
        'single_rejection_rate': single_rejection_rate,
        'hit': hit,
        'miss': miss,
        'acceptance_accuracy': acceptance_accuracy,
        'performance_loss': performance_loss
    }
    
    # Metriche per Voting 2/2
    VOTING_2OF2_METRICS = {
        'double_correct_prediction_rate': double_correct_prediction_rate,
        'double_wrong_prediction_rate': double_wrong_prediction_rate,
        'double_rejection_rate': double_rejection_rate
    }
    
    # Metriche per Recovery Block
    RECOVERY_BLOCK_METRICS = {
        'recovery_rate': recovery_rate,
        'recovery_failure_rate': recovery_failure_rate,
        'recovery_rejection_rate': recovery_rejection_rate,
        'recovery_correct_prediction_rate': recovery_correct_prediction_rate,
        'recovery_wrong_prediction_rate': recovery_wrong_prediction_rate,
        'recovery_rejection_prediction_rate': recovery_rejection_prediction_rate
    }
    
    # Metriche per Majority Voting
    MAJORITY_VOTING_METRICS = {
        'majority_voting_correct_prediction_rate': majority_voting_correct_prediction_rate,
        'majority_voting_wrong_prediction_rate': majority_voting_wrong_prediction_rate,
        'majority_voting_rejection_prediction_rate': majority_voting_rejection_prediction_rate
    }
    
    # Metriche per Voting 1/N
    VOTING_1OFN_METRICS = {
        'single_vote_correct_prediction_rate': single_vote_correct_prediction_rate,
        'single_vote_wrong_prediction_rate': single_vote_wrong_prediction_rate,
        'single_vote_rejection_rate': single_vote_rejection_rate,
        'ideal_single_vote_rate': ideal_single_vote_rate
    }
    
    # Metriche che richiedono X_test e model per predict_proba
    METRICS_REQUIRING_MODEL = {
        'single_correct_prediction_rate', 'single_misclassification_rate',
        'single_rejection_rate', 'hit', 'miss', 'acceptance_accuracy', 'performance_loss',
        'recovery_correct_prediction_rate', 'recovery_wrong_prediction_rate',
        'recovery_rejection_prediction_rate'
    }
    
    def get_metrics_for_model(self, model: Union[Ensemble, abstract_rejection_decorator]) -> Dict[str, Type]:
        """Determina quali metriche calcolare in base al tipo di modello."""
        metrics = {}
        
        if isinstance(model, Voting2of2Ensemble):
            metrics.update(self.CLASSIC_METRICS)
            metrics.update(self.VOTING_2OF2_METRICS)
        
        elif isinstance(model, RecoveryBlockEnsemble):
            metrics.update(self.CLASSIC_METRICS)
            metrics.update(self.RECOVERY_BLOCK_METRICS)
        
        elif isinstance(model, MajorityVotingEnsemble):
            metrics.update(self.CLASSIC_METRICS)
            metrics.update(self.MAJORITY_VOTING_METRICS)
        
        elif isinstance(model, Voting1ofNEnsemble):
            metrics.update(self.CLASSIC_METRICS)
            metrics.update(self.VOTING_1OFN_METRICS)
        
        elif isinstance(model, abstract_rejection_decorator):
            metrics.update(self.SINGLE_METRICS)
        
        else:
            raise ValueError(f"Tipo di modello non supportato: {type(model)}")
        
        return metrics
    
    def calculate(self, ds: dataset, model: Union[Ensemble, abstract_rejection_decorator],
                  metrics: List[str] = None) -> Dict[str, float]:
        """
        Calcola le metriche per un modello (ensemble o singolo classificatore).
        
        Args:
            ds: Dataset preparato
            model: Ensemble o classificatore con rejection
            metrics: Lista di metriche da calcolare (None = tutte quelle appropriate)
        
        Returns:
            Dizionario {nome_metrica: valore}
        """
        X_train, X_test, y_train, y_test = ds.data
        
        # Training
        model.train(X_train, y_train)
        
        # Determina metriche disponibili per questo modello
        available_metrics = self.get_metrics_for_model(model)
        
        # Se non specificate, calcola tutte
        if metrics is None:
            metrics_to_compute = available_metrics
        else:
            metrics_to_compute = {k: v for k, v in available_metrics.items() if k in metrics}
        
        # Ottieni predizioni
        if isinstance(model, Ensemble):
            predictions = model.get_predictions(X_test)
        else:
            # Singolo classificatore: shape (1, n_samples)
            predictions = model.predict(X_test).reshape(1, -1)
        
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        predictions = predictions.astype(str)
        y_test_array = y_test_array.astype(str)
        
        # Calcola metriche
        results = {}
        for metric_name, metric_class in metrics_to_compute.items():
            value = self._compute_metric(
                metric_name, metric_class, predictions, y_test_array, X_test, model
            )
            results[metric_name] = value
        
        return results
    
    def _compute_metric(self, metric_name: str, metric_class: Type,
                        predictions: np.ndarray, y_test: np.ndarray,
                        X_test: np.ndarray, model) -> float:
        """Calcola una singola metrica."""
        try:
            metric_instance = metric_class()
            
            # Metriche singolo classificatore
            if metric_name in self.SINGLE_METRICS:
                return metric_instance._compute(predictions[0], y_test, X_test, model)
            
            # Metriche recovery composite (richiedono lista classificatori)
            elif metric_name in ['recovery_correct_prediction_rate',
                                 'recovery_wrong_prediction_rate',
                                 'recovery_rejection_prediction_rate']:
                return metric_instance._compute(predictions, y_test, X_test, model.classifiers)
            
            # Altre metriche (classiche e specifiche ensemble)
            else:
                return metric_instance._compute(predictions, y_test)
        
        except Exception as e:
            print(f"Errore calcolo {metric_name}: {e}")
            return None