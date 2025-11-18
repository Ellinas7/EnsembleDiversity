import numpy as np
from typing import Union, Optional

class RejectionClassifier:
    """
    Wrapper che aggiunge rejection option a qualsiasi classificatore.
    
    Se la confidenza più alta è sotto la soglia del 90%, predice "I don't know" (-1).
    """
    
    def __init__(self, base_model, confidence_threshold: float = 0.9):
        """
        Args:
            base_model: Classificatore che implementa predict_proba()
            confidence_threshold: Soglia minima di confidenza (default 0.9 = 90%)
        """
        self.base_model = base_model
        self.confidence_threshold = confidence_threshold
        self.rejection_label = -1  # Label per "I don't know"

    def train(self, X_train, y_train):
        """Addestra il modello base"""
        self.base_model.train(X_train, y_train)

    def predict(self, X_test):
        """
        Predice con rejection option.
        
        Returns:
            Array con predizioni: classi originali o -1 (I don't know)
        """
        probas = self.predict_proba(X_test)
        
        # Confidenza massima per ogni campione
        max_confidences = np.max(probas, axis=1)
        
        # Predizioni del modello base
        base_predictions = self.base_model.predict(X_test)
        
        # Applica rejection: se confidenza < soglia → -1
        predictions = np.where(
            max_confidences >= self.confidence_threshold,
            base_predictions,
            self.rejection_label
        )
        
        return predictions
    
    def predict_proba(self, X_test):
        """Restituisce le probabilità del modello base"""
        return self.base_model.model.predict_proba(X_test)
    
    def get_confidence_scores(self, X_test):
        """Restituisce gli score di confidenza per ogni campione"""
        probas = self.predict_proba(X_test)
        return np.max(probas, axis=1)
    
    def get_rejection_stats(self, X_test):
        """Statistiche sui campioni rifiutati"""
        predictions = self.predict(X_test)
        n_total = len(predictions)
        n_rejected = np.sum(predictions == self.rejection_label)
        n_accepted = n_total - n_rejected
        
        return {
            'total_samples': n_total,
            'accepted': n_accepted,
            'rejected': n_rejected,
            'coverage': n_accepted / n_total,  # Percentuale di campioni classificati
            'rejection_rate': n_rejected / n_total
        }
    
    