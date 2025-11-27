"""
my_experiments.py - Script per popolare il Megafile.csv con tutti gli esperimenti

Uso dell'OPZIONE C: import come modulo con namespace 'el.'

Questo script usa un singolo import:
    import experiment_logger as el

E accede a tutto tramite namespace el.:
    - el.ExperimentLogger()
    - el.get_decision_tree()
    - el.wrap_static_threshold()
    - el.METRICHE_SINGOLO
    - el.DATASETS

Aggiungi i tuoi esperimenti nel main() e esegui:
    python my_experiments.py
"""

import experiment_logger as el


# =============================================================================
# MAIN - Aggiungi qui i tuoi esperimenti
# =============================================================================

def main():
    """
    Main script - Aggiungi chiamate a el.logger.run_experiment() qui.
    
    Pattern:
        1. Crea algoritmo base con el.get_*()
        2. Wrappa con rejection decorator usando el.wrap_*()
        3. Chiama logger.run_experiment()
    """
    
    print("\n" + "="*80)
    print("INIZIO ESPERIMENTI")
    print("="*80 + "\n")
    
    # Inizializza logger (una volta sola)
    logger = el.ExperimentLogger("/Users/matteopascuzzo/Desktop/Megafile.csv")
    
    # =========================================================================
    # ESPERIMENTO 1: DecisionTree + StaticThreshold 0.9
    # =========================================================================
    
    dt = el.get_decision_tree()
    dt_rej = el.wrap_static_threshold(dt, threshold=0.9)
    
    logger.run_experiment(
        dataset_path=el.DATASETS['arancino_all_scikit'],
        dataset_name='arancino_all_scikit',
        ml_algorithm=dt_rej,
        experiment_name='arancinoallscikit_decisiontree_staticthreshold',
        metrics_to_calculate=el.METRICHE_SINGOLO
    )
    
    """logger.delete_experiment(1)"""
    


    # =========================================================================
    # FINE ESPERIMENTI
    # =========================================================================
    
    print("\n" + "="*80)
    print("✓ TUTTI GLI ESPERIMENTI COMPLETATI")
    print("✓ Risultati salvati in: /Users/matteopascuzzo/Desktop/Megafile.csv")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()