"""
experiment_logger.py - Sistema completo per gestione esperimenti ML con rejection option

Questo modulo fornisce:
- ExperimentLogger: classe per logging e salvataggio risultati
- Factory functions: get_decision_tree(), get_random_forest(), get_xgboost(), get_adaboost(),
                    get_extra_trees(), get_gradient_boosting(), get_lightgbm(), get_catboost(),
                    get_rotation_forest(), get_random_rotation_forest()
- Wrapper functions: wrap_static_threshold(), wrap_percentile_threshold()
- Costanti: METRICHE_SINGOLO, METRICHE_DIVERSITY, METRICHE_DOPPIO, METRICHE_TERNA
- DATASETS: dizionario paths dataset comuni

Uso:
    import experiment_logger as el
    
    logger = el.ExperimentLogger("/path/to/Megafile.csv")
    dt = el.get_decision_tree()
    dt_rej = el.wrap_static_threshold(dt, 0.9)
    logger.run_experiment(..., metrics_to_calculate=el.METRICHE_SINGOLO)
"""

import pandas as pd
import numpy as np
from pathlib import Path

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


# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

# Parametri comuni
RANDOM_STATE = 42

# Dataset paths (modifica questi path secondo la tua struttura)
DATASETS = {
    'arancino_all_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/arancino_all_scikit.csv',
    'mafaulda': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/MAFAULDA.csv',
    'mechfailure_electriccomponent_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/MechFailure_ElectricComponent_scikit.csv',
    'scaniatrucks_aps_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/ScaniaTrucks_APS_scikit.csv',
    'backblaze_2017_5percrate_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/BackBlaze_2017_5PercRate_scikit.csv',
    'backblaze_2023': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/BackBlaze_2023.csv',
    'baidu_smartdataset_15perc_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/Baidu_SMART Dataset_15Perc_scikit.csv',
    'baiot_mirai': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/BAIoT_mirai.csv',
    'full_iot_ids_dataset_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/Full_IoT_IDS_Dataset_scikit.csv',
    'iscx_meta': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/ISCX_Meta.csv'
}


# =============================================================================
# COSTANTI METRICHE - Divise per categoria
# =============================================================================

METRICHE_SINGOLO = [
    'single_correct_prediction_rate',
    'single_misclassification_rate',
    'single_rejection_rate',
    'hit',
    'miss',
    'acceptance_accuracy',
    'performance_loss'
]

METRICHE_DOPPIO = [
    'double_correct_prediction_rate',
    'double_wrong_prediction_rate',
    'double_rejection_rate',
    'recovery_rate',
    'recovery_failure_rate',
    'recovery_rejection_rate',
    'recovery_correct_prediction_rate',
    'recovery_wrong_prediction_rate',
    'recovery_rejection_prediction_rate'
]

METRICHE_TERNA = [
    'majority_voting_correct_prediction_rate',
    'majority_voting_wrong_prediction_rate',
    'majority_voting_rejection_prediction_rate',
    'single_vote_correct_prediction_rate',
    'single_vote_wrong_prediction_rate',
    'single_vote_rejection_rate',
    'ideal_single_vote_rate'
]

METRICHE_DIVERSITY = [
    'q_statistic',
    'disagreement_measure',
    'double_fault_measure',
    'entropy_measure',
    'kohavi_wolpert_variance',
    'generalized_diversity',
    'coincident_failure_diversity'
]

# Set completo di tutte le metriche
METRICHE_COMPLETE = METRICHE_SINGOLO + METRICHE_DOPPIO + METRICHE_TERNA + METRICHE_DIVERSITY

# =============================================================================
# ORDINE COLONNE CSV - Ordine fisso per tutte le colonne del Megafile.csv
# =============================================================================

# Colonne base 
_BASE_COLUMNS = ['experiment_number', 'experiment_name', 'dataset_name', 'classification_strategy']

# Colonne metriche nell'ordine specificato
_ORDERED_METRIC_COLUMNS = (
    # 1. DIVERSITY (7 metriche)
    METRICHE_DIVERSITY +
    
    # 2. SINGOLO (7 metriche)
    METRICHE_SINGOLO +
    
    # 3. DOPPIO - VOTING 2/2 (3 metriche)
    [
        'double_correct_prediction_rate',
        'double_wrong_prediction_rate',
        'double_rejection_rate',
    ] +
    
    # 4. DOPPIO - RECOVERY BLOCK (6 metriche)
    [
        'recovery_rate',
        'recovery_failure_rate',
        'recovery_rejection_rate',
        'recovery_correct_prediction_rate',
        'recovery_wrong_prediction_rate',
        'recovery_rejection_prediction_rate',
    ] +
    
    # 5. TERNA - MAJORITY VOTING (3 metriche)
    [
        'majority_voting_correct_prediction_rate',
        'majority_voting_wrong_prediction_rate',
        'majority_voting_rejection_prediction_rate',
    ] +
    
    # 6. TERNA - VOTING 1/N (4 metriche)
    [
        'single_vote_correct_prediction_rate',
        'single_vote_wrong_prediction_rate',
        'single_vote_rejection_rate',
        'ideal_single_vote_rate',
    ]
)

# Lista completa ordinata di tutte le colonne
ORDERED_COLUMNS = _BASE_COLUMNS + _ORDERED_METRIC_COLUMNS


# =============================================================================
# FACTORY FUNCTIONS - Algoritmi ML (istanze fresche)
# =============================================================================

def get_decision_tree(random_state=None):
    """
    Crea nuova istanza di DecisionTree
    
    Args:
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
    
    Returns:
        Istanza di decision_tree
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return decision_tree(random_state=random_state)


def get_random_forest(n_estimators, random_state=None):
    """
    Crea nuova istanza di RandomForest
    
    Args:
        n_estimators: Numero di alberi (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
    
    Returns:
        Istanza di random_forest
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return random_forest(n_estimators=n_estimators, random_state=random_state)


def get_xgboost(n_estimators, random_state=None):
    """
    Crea nuova istanza di XGBoost
    
    Args:
        n_estimators: Numero di alberi (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
    
    Returns:
        Istanza di xgboost
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return xgboost(n_estimators=n_estimators, random_state=random_state)


def get_adaboost(n_estimators, random_state=None):
    """
    Crea nuova istanza di AdaBoost
    
    Args:
        n_estimators: Numero di estimatori (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
    
    Returns:
        Istanza di adaboost
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return adaboost(n_estimators=n_estimators, random_state=random_state)


def get_extra_trees(n_estimators, random_state=None):
    """
    Crea nuova istanza di ExtraTrees
    
    Args:
        n_estimators: Numero di alberi (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
    
    Returns:
        Istanza di extra_trees
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return extra_trees(n_estimators=n_estimators, random_state=random_state)


def get_gradient_boosting(n_estimators, random_state=None, max_depth=3, 
                         learning_rate=0.1, subsample=1.0):
    """
    Crea nuova istanza di Gradient Boosting Decision Trees (GBDT)
    
    Args:
        n_estimators: Numero di alberi (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
        max_depth: Profondità massima degli alberi (default: 3)
        learning_rate: Tasso di apprendimento (default: 0.1)
        subsample: Frazione di campioni per fit (default: 1.0)
    
    Returns:
        Istanza di gradient_boosting_decision_trees
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return gradient_boosting_decision_trees(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample
    )


def get_lightgbm(n_estimators, random_state=None, max_depth=-1, 
                learning_rate=0.1, num_leaves=31):
    """
    Crea nuova istanza di LightGBM
    
    Args:
        n_estimators: Numero di alberi (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
        max_depth: Profondità massima degli alberi (default: -1 = no limit)
        learning_rate: Tasso di apprendimento (default: 0.1)
        num_leaves: Numero massimo di foglie per albero (default: 31)
    
    Returns:
        Istanza di light_gbm
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return light_gbm(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves
    )


def get_catboost(n_estimators, random_state=None, max_depth=6, 
                learning_rate=0.1):
    """
    Crea nuova istanza di CatBoost
    
    Args:
        n_estimators: Numero di iterazioni (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
        max_depth: Profondità massima degli alberi (default: 6)
        learning_rate: Tasso di apprendimento (default: 0.1)
    
    Returns:
        Istanza di catboost
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return catboost(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        learning_rate=learning_rate
    )


def get_rotation_forest(n_estimators, random_state=None, n_jobs=-1,
                       max_features=None, bootstrap=True):
    """
    Crea nuova istanza di Rotation Forest
    
    Args:
        n_estimators: Numero di alberi (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
        n_jobs: Numero di job paralleli (default: -1 = tutti i core)
        max_features: Numero massimo di features per split (default: None)
        bootstrap: Se usare bootstrap sampling (default: True)
    
    Returns:
        Istanza di rotation_forest
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return rotation_forest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_features=max_features,
        bootstrap=bootstrap
    )


def get_random_rotation_forest(n_estimators, random_state=None, n_jobs=-1,
                              max_features=None, bootstrap=True):
    """
    Crea nuova istanza di Random Rotation Forest
    
    Args:
        n_estimators: Numero di alberi (OBBLIGATORIO)
        random_state: Seed per riproducibilità (default: RANDOM_STATE globale)
        n_jobs: Numero di job paralleli (default: -1 = tutti i core)
        max_features: Numero massimo di features per split (default: None)
        bootstrap: Se usare bootstrap sampling (default: True)
    
    Returns:
        Istanza di random_rotation_forest
    """
    if random_state is None:
        random_state = RANDOM_STATE
    return random_rotation_forest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_features=max_features,
        bootstrap=bootstrap
    )


# =============================================================================
# WRAPPER FUNCTIONS - Rejection Decorators
# =============================================================================

def wrap_static_threshold(algorithm, threshold=0.9):
    """
    Wrappa algoritmo con StaticThreshold rejection decorator
    
    Args:
        algorithm: Istanza dell'algoritmo ML da wrappare
        threshold: Soglia di confidenza (default: 0.9 = 90%)
    
    Returns:
        Algoritmo wrappato con rejection capability
    """
    return static_threshold_rejection_decorator(
        base_algorithm=algorithm,
        confidence_threshold=threshold
    )


def wrap_percentile_threshold(algorithm, percentile=10.0):
    """
    Wrappa algoritmo con PercentileThreshold rejection decorator
    
    Args:
        algorithm: Istanza dell'algoritmo ML da wrappare
        percentile: Percentile di rejection (default: 10.0 = 10%)
    
    Returns:
        Algoritmo wrappato con rejection capability
    """
    return percentile_threshold_rejection_decorator(
        base_algorithm=algorithm,
        rejection_percentile=percentile
    )


# =============================================================================
# CLASSE EXPERIMENTLOGGER
# =============================================================================

class ExperimentLogger:
    """
    Classe per gestire gli esperimenti e salvare i risultati nel Megafile.csv
    
    Include TUTTE le metriche disponibili nel progetto.
    """
    
    def __init__(self, csv_path: str = "Megafile.csv"):
        """
        Args:
            csv_path: Path del file CSV dove salvare i risultati
        """
        self.csv_path = Path(csv_path)
        
        # =========================================================================
        # IMPORTANTE: Definisci metrics_map PRIMA di caricare il CSV
        # perché _load_or_create_csv() ne ha bisogno per creare tutte le colonne
        # =========================================================================
        
        # Mappa completa: nome metrica -> istanza metrica
        self.metrics_map = {
            # SINGOLO CLASSIFICATORE (7 metriche)
            'single_correct_prediction_rate': single_correct_prediction_rate(),
            'single_misclassification_rate': single_misclassification_rate(),
            'single_rejection_rate': single_rejection_rate(),
            'hit': hit(),
            'miss': miss(),
            'acceptance_accuracy': acceptance_accuracy(),
            'performance_loss': performance_loss(),
            
            # COPPIA - VOTING 2/2 (3 metriche)
            'double_correct_prediction_rate': double_correct_prediction_rate(),
            'double_wrong_prediction_rate': double_wrong_prediction_rate(),
            'double_rejection_rate': double_rejection_rate(),
            
            # COPPIA - RECOVERY BLOCK (6 metriche)
            'recovery_rate': recovery_rate(),
            'recovery_failure_rate': recovery_failure_rate(),
            'recovery_rejection_rate': recovery_rejection_rate(),
            'recovery_correct_prediction_rate': recovery_correct_prediction_rate(),
            'recovery_wrong_prediction_rate': recovery_wrong_prediction_rate(),
            'recovery_rejection_prediction_rate': recovery_rejection_prediction_rate(),
            
            # TERNA - MAJORITY VOTING (3 metriche)
            'majority_voting_correct_prediction_rate': majority_voting_correct_prediction_rate(),
            'majority_voting_wrong_prediction_rate': majority_voting_wrong_prediction_rate(),
            'majority_voting_rejection_prediction_rate': majority_voting_rejection_prediction_rate(),
            
            # TERNA - VOTING 1/N (4 metriche)
            'single_vote_correct_prediction_rate': single_vote_correct_prediction_rate(),
            'single_vote_wrong_prediction_rate': single_vote_wrong_prediction_rate(),
            'single_vote_rejection_rate': single_vote_rejection_rate(),
            'ideal_single_vote_rate': ideal_single_vote_rate(),
            
            # METRICHE CLASSICHE DI DIVERSITY (7 metriche)
            'q_statistic': Q_statistic(),
            'disagreement_measure': disagreement_measure(),
            'double_fault_measure': double_fault_measure(),
            'entropy_measure': entropy_measure(),
            'kohavi_wolpert_variance': kohavi_wolpert_variance(),
            'generalized_diversity': generalized_diversity(),
            'coincident_failure_diversity': coincident_failure_diversity(),
        }
        
        print(f"✓ ExperimentLogger inizializzato con {len(self.metrics_map)} metriche disponibili")
        
        # Ora carica/crea CSV (usa self.metrics_map)
        self.df = self._load_or_create_csv()
    
    def _load_or_create_csv(self) -> pd.DataFrame:
        """Carica il CSV esistente o ne crea uno nuovo con TUTTE le colonne nell'ordine corretto"""
        if self.csv_path.exists():
            print(f"✓ Caricato {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            # Assicurati che tutte le colonne esistano
            for col in ORDERED_COLUMNS:
                if col not in df.columns:
                    if col == 'experiment_number':
                        # Se manca experiment_number, aggiungi basandosi sull'ordine attuale
                        df.insert(0, 'experiment_number', range(1, len(df) + 1))
                    else:
                        df[col] = "non calcolata per questo esperimento"
            
            # Riordina colonne secondo ORDERED_COLUMNS (mantieni solo colonne valide)
            existing_ordered_cols = [col for col in ORDERED_COLUMNS if col in df.columns]
            df = df[existing_ordered_cols]
            
            return df
        else:
            print(f"✓ Creato nuovo {self.csv_path}")
            # Crea DataFrame con TUTTE le colonne nell'ordine corretto
            return pd.DataFrame(columns=ORDERED_COLUMNS)
           
    
    def run_experiment(self, 
                      dataset_path: str,
                      dataset_name: str,
                      ml_algorithm,
                      experiment_name: str,
                      metrics_to_calculate: list):
        """
        Esegue un esperimento completo e salva i risultati.
        
        Args:
            dataset_path: Path del dataset CSV
            dataset_name: Nome del dataset
            ml_algorithm: Istanza dell'algoritmo ML (già wrappato con rejection decorator)
            experiment_name: Nome univoco dell'esperimento
            metrics_to_calculate: Lista di nomi delle metriche da calcolare
        """
        print(f"\n{'='*80}")
        print(f"ESPERIMENTO: {experiment_name}")
        print(f"{'='*80}")
        
        # 1. Carica e prepara dataset
        ds = dataset(
            file_path=dataset_path,
            dataset_name=dataset_name,
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        ds.preprocess()
        X_train, X_test, y_train, y_test = ds.data
        
        # 2. Addestra il modello
        ml_algorithm.train(X_train, y_train)
        
        # 3. Ottieni predizioni (con rejection)
        predictions = ml_algorithm.predict(X_test)
        
        print(f"\n{'='*80}")
        print("CALCOLO METRICHE")
        print(f"{'='*80}")
        
        # 4. Inizializza results con TUTTE le colonne
        # Determina experiment_number
        mask = self.df['experiment_name'] == experiment_name
        if mask.any():
            # Esperimento esiste già → mantieni stesso numero
            exp_number = self.df.loc[mask, 'experiment_number'].iloc[0]
        else:
            # Nuovo esperimento → assegna prossimo numero disponibile
            if len(self.df) == 0:
                exp_number = 1
            else:
                exp_number = self.df['experiment_number'].max() + 1
        
        results = {
            'experiment_number': exp_number,
            'experiment_name': experiment_name,
            'dataset_name': dataset_name,
            'classification_strategy': ml_algorithm.name
        }
        
        # Inizializza TUTTE le metriche con "non calcolata per questo esperimento"
        for metric_name in self.metrics_map.keys():
            results[metric_name] = "non calcolata per questo esperimento"
        
        # 5. Calcola solo le metriche specificate (sovrascrive il valore sopra)
        for metric_name in metrics_to_calculate:
            if metric_name not in self.metrics_map:
                print(f"⚠ Metrica '{metric_name}' non trovata, saltata")
                print(f"   Usa logger.print_available_metrics() per vedere la lista completa")
                continue
            
            try:
                metric_obj = self.metrics_map[metric_name]
                value = metric_obj._compute(predictions, y_test, X_test, ml_algorithm)
                results[metric_name] = value
                print(f"  {metric_name}: {value:.6f}")
            except Exception as e:
                print(f"⚠ Errore nel calcolo di '{metric_name}': {e}")
                results[metric_name] = None
        
        # 6. Aggiorna DataFrame
        # Controlla se l'esperimento esiste già
        mask = self.df['experiment_name'] == experiment_name
        if mask.any():
            # Aggiorna riga esistente
            for col, val in results.items():
                if col not in self.df.columns:
                    # Aggiungi colonna se non esiste (inizializza con placeholder)
                    self.df[col] = "non calcolata per questo esperimento"
                self.df.loc[mask, col] = val
            print(f"\n✓ Esperimento '{experiment_name}' aggiornato")
        else:
            # Aggiungi nuova riga
            # Assicurati che il DataFrame abbia tutte le colonne
            for col in results.keys():
                if col not in self.df.columns:
                    self.df[col] = "non calcolata per questo esperimento"
            
            new_row = pd.DataFrame([results])
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            print(f"\n✓ Esperimento '{experiment_name}' aggiunto")
        
        # 7. Riordina colonne secondo ORDERED_COLUMNS e salva CSV
        # Mantieni solo le colonne che esistono nel DataFrame
        existing_ordered_cols = [col for col in ORDERED_COLUMNS if col in self.df.columns]
        self.df = self.df[existing_ordered_cols]
        
        self.df.to_csv(self.csv_path, index=False)
        print(f"✓ Salvato in {self.csv_path}")
        print(f"{'='*80}\n")
        
        return results
    
    def delete_experiment(self, number: int):
        """
        Elimina un esperimento per numero e rinumera automaticamente gli esperimenti successivi.
        
        Args:
            number: Numero dell'esperimento da eliminare (experiment_number)
        
        Esempio:
            logger.delete_experiment(5)  # Elimina esperimento #5, rinumera 6→5, 7→6, etc.
        """
        # Verifica che il numero esista
        if number not in self.df['experiment_number'].values:
            print(f"⚠ Esperimento #{number} non trovato nel CSV")
            print(f"   Esperimenti disponibili: {sorted(self.df['experiment_number'].tolist())}")
            return
        
        # Trova nome esperimento per feedback
        exp_name = self.df.loc[self.df['experiment_number'] == number, 'experiment_name'].iloc[0]
        
        print(f"\n{'='*80}")
        print(f"ELIMINAZIONE ESPERIMENTO #{number}")
        print(f"{'='*80}")
        print(f"Nome: {exp_name}")
        
        # Elimina riga
        self.df = self.df[self.df['experiment_number'] != number].reset_index(drop=True)
        print(f"✓ Esperimento #{number} eliminato")
        
        # Rinumera esperimenti successivi
        # Tutti gli esperimenti con numero > number vengono decrementati di 1
        self.df.loc[self.df['experiment_number'] > number, 'experiment_number'] -= 1
        
        # Riordina per numero (dovrebbe già essere ordinato, ma per sicurezza)
        self.df = self.df.sort_values('experiment_number').reset_index(drop=True)
        
        print(f"✓ Esperimenti rinumerati (#{number+1}→#{number}, #{number+2}→#{number+1}, ...)")
        
        # Salva CSV
        existing_ordered_cols = [col for col in ORDERED_COLUMNS if col in self.df.columns]
        self.df = self.df[existing_ordered_cols]
        self.df.to_csv(self.csv_path, index=False)
        print(f"✓ Salvato in {self.csv_path}")
        print(f"{'='*80}\n")