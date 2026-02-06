"""
Valuta se un ensemble è migliorativo rispetto ai base learner.
Adattato per i 16 file generati dal nuovo main.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


class EnsembleEvaluator:
    """
    Valuta se un ensemble è migliorativo rispetto ai base learner.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)
        self.n_classifiers = self._detect_n_classifiers()
    
    def _detect_n_classifiers(self) -> int:
        """Rileva se ci sono 2 o 3 base learner"""
        if 'classifier_3_correct_rate' in self.df.columns:
            return 3
        return 2
    
    def _get_base_metrics(self, row) -> list:
        """Restituisce lista di tuple (correct_rate, misclassification_rate) per ogni base learner"""
        metrics = []
        for i in range(1, self.n_classifiers + 1):
            metrics.append((
                row[f'classifier_{i}_correct_rate'],
                row[f'classifier_{i}_misclassification_rate']
            ))
        return metrics
    
    def ensemble_wins(self, row) -> bool:
        """Determina se l'ensemble vince per una singola riga"""
        ens_cr = row['ensemble_correct_rate']
        ens_mr = row['ensemble_misclassification_rate']
        
        base_metrics = self._get_base_metrics(row)
        base_mrs = [m[1] for m in base_metrics]
        
        min_base_mr = min(base_mrs)
        
        # Caso 1: ensemble ha MR strettamente minore di tutti
        if ens_mr < min_base_mr:
            # best_base_cr: il CR più alto tra i base learner con MR minimo
            best_base_cr = max(cr for cr, mr in base_metrics if mr == min_base_mr)
            return ens_cr >= best_base_cr - 0.05
        
        # Caso 2: ensemble ha MR minimo a pari merito
        if ens_mr == min_base_mr:
            # Trova i correct rate dei base learner con MR minimo
            tied_crs = [cr for cr, mr in base_metrics if mr == ens_mr]
            
            # Ensemble vince se ha CR strettamente maggiore di tutti quelli a parità
            return ens_cr > max(tied_crs)
        
        return False
    
    def evaluate(self) -> pd.DataFrame:
        """Valuta tutte le righe e aggiunge colonna 'ensemble_wins'"""
        self.df['ensemble_wins'] = self.df.apply(self.ensemble_wins, axis=1)
        return self.df
    
    def save(self, output_path: str = None):
        """Salva il DataFrame con la nuova colonna e il totale vittorie"""
        if output_path is None:
            output_path = self.file_path.replace('.xlsx', '_evaluated.xlsx')
        
        # Crea copia per non modificare l'originale
        df_to_save = self.df.copy()
        
        # Aggiungi riga con totale vittorie
        total_row = {col: '' for col in df_to_save.columns}
        total_row['ensemble_wins'] = df_to_save['ensemble_wins'].sum()
        df_to_save = pd.concat([df_to_save, pd.DataFrame([total_row])], ignore_index=True)
        
        df_to_save.to_excel(output_path, index=False)
        return output_path
    
    def get_stats(self) -> dict:
        """Restituisce statistiche sui risultati"""
        if 'ensemble_wins' not in self.df.columns:
            self.evaluate()
        
        total = len(self.df)
        wins = self.df['ensemble_wins'].sum()
        
        return {
            'total': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0
        }


# === PATH DEI FILE ===

RESULTS_DIR = Path("/Users/matteopascuzzo/Desktop/Results")
OUTPUT_DIR = Path("/Users/matteopascuzzo/Desktop/Results/Evaluated")

FILES = {
    # Coppie Static - Gruppo 1
    'CoppieStaticGruppo1_Voting2su2': RESULTS_DIR / "CoppieStaticGruppo1_Voting2su2.xlsx",
    'CoppieStaticGruppo1_RecoveryBlock': RESULTS_DIR / "CoppieStaticGruppo1_RecoveryBlock.xlsx",
    
    # Coppie Static - Gruppo 2
    'CoppieStaticGruppo2_Voting2su2': RESULTS_DIR / "CoppieStaticGruppo2_Voting2su2.xlsx",
    'CoppieStaticGruppo2_RecoveryBlock': RESULTS_DIR / "CoppieStaticGruppo2_RecoveryBlock.xlsx",
    
    # Coppie Percentile - Gruppo 1
    'CoppiePercentileGruppo1_Voting2su2': RESULTS_DIR / "CoppiePercentileGruppo1_Voting2su2.xlsx",
    'CoppiePercentileGruppo1_RecoveryBlock': RESULTS_DIR / "CoppiePercentileGruppo1_RecoveryBlock.xlsx",
    
    # Coppie Percentile - Gruppo 2
    'CoppiePercentileGruppo2_Voting2su2': RESULTS_DIR / "CoppiePercentileGruppo2_Voting2su2.xlsx",
    'CoppiePercentileGruppo2_RecoveryBlock': RESULTS_DIR / "CoppiePercentileGruppo2_RecoveryBlock.xlsx",
    
    # Triple Static - Gruppo 1
    'TripleStaticGruppo1_MajorityVoting': RESULTS_DIR / "TripleStaticGruppo1_MajorityVoting.xlsx",
    'TripleStaticGruppo1_Voting1suN': RESULTS_DIR / "TripleStaticGruppo1_Voting1suN.xlsx",
    
    # Triple Static - Gruppo 2
    'TripleStaticGruppo2_MajorityVoting': RESULTS_DIR / "TripleStaticGruppo2_MajorityVoting.xlsx",
    'TripleStaticGruppo2_Voting1suN': RESULTS_DIR / "TripleStaticGruppo2_Voting1suN.xlsx",
    
    # Triple Percentile - Gruppo 1
    'TriplePercentileGruppo1_MajorityVoting': RESULTS_DIR / "TriplePercentileGruppo1_MajorityVoting.xlsx",
    'TriplePercentileGruppo1_Voting1suN': RESULTS_DIR / "TriplePercentileGruppo1_Voting1suN.xlsx",
    
    # Triple Percentile - Gruppo 2
    'TriplePercentileGruppo2_MajorityVoting': RESULTS_DIR / "TriplePercentileGruppo2_MajorityVoting.xlsx",
    'TriplePercentileGruppo2_Voting1suN': RESULTS_DIR / "TriplePercentileGruppo2_Voting1suN.xlsx",
}


def evaluate_all(save_files: bool = True):
    """Valuta tutti i 16 file e stampa statistiche"""
    print("=" * 70)
    print("VALUTAZIONE ENSEMBLE")
    print("=" * 70)
    
    if save_files:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for name, path in FILES.items():
        try:
            evaluator = EnsembleEvaluator(str(path))
            evaluator.evaluate()
            stats = evaluator.get_stats()
            results[name] = stats
            
            if save_files:
                output_path = OUTPUT_DIR / f"{name}_evaluated.xlsx"
                evaluator.save(str(output_path))
            
            print(f"\n{name}:")
            print(f"  Vittorie: {stats['wins']}/{stats['total']} ({stats['win_rate']*100:.1f}%)")
            
        except Exception as e:
            print(f"\n{name}: ERRORE - {e}")
    
    # Riepilogo per categoria
    print("\n" + "=" * 70)
    print("RIEPILOGO PER CATEGORIA")
    print("=" * 70)
    
    categories = {
        'Coppie Static Gruppo 1': ['CoppieStaticGruppo1_Voting2su2', 'CoppieStaticGruppo1_RecoveryBlock'],
        'Coppie Static Gruppo 2': ['CoppieStaticGruppo2_Voting2su2', 'CoppieStaticGruppo2_RecoveryBlock'],
        'Coppie Percentile Gruppo 1': ['CoppiePercentileGruppo1_Voting2su2', 'CoppiePercentileGruppo1_RecoveryBlock'],
        'Coppie Percentile Gruppo 2': ['CoppiePercentileGruppo2_Voting2su2', 'CoppiePercentileGruppo2_RecoveryBlock'],
        'Triple Static Gruppo 1': ['TripleStaticGruppo1_MajorityVoting', 'TripleStaticGruppo1_Voting1suN'],
        'Triple Static Gruppo 2': ['TripleStaticGruppo2_MajorityVoting', 'TripleStaticGruppo2_Voting1suN'],
        'Triple Percentile Gruppo 1': ['TriplePercentileGruppo1_MajorityVoting', 'TriplePercentileGruppo1_Voting1suN'],
        'Triple Percentile Gruppo 2': ['TriplePercentileGruppo2_MajorityVoting', 'TriplePercentileGruppo2_Voting1suN'],
    }
    
    for cat_name, file_names in categories.items():
        cat_wins = sum(results[f]['wins'] for f in file_names if f in results)
        cat_total = sum(results[f]['total'] for f in file_names if f in results)
        if cat_total > 0:
            print(f"\n{cat_name}: {cat_wins}/{cat_total} ({cat_wins/cat_total*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("RIEPILOGO TOTALE")
    print("=" * 70)
    
    total_wins = sum(r['wins'] for r in results.values())
    total_rows = sum(r['total'] for r in results.values())
    
    print(f"\nTotale: {total_wins}/{total_rows} ({total_wins/total_rows*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    evaluate_all(save_files=True)