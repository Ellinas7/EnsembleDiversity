import numpy as np
import pandas as pd
from pathlib import Path

# Import dataset
from datasets.dataset import dataset

# Import Decision Tree
from ML_algorithms.decision_tree import decision_tree

# Import Rejection Classifier
from ML_algorithms.rejection_classifier import RejectionClassifier


def create_detailed_predictions_table(rejection_model, X_test, y_test, n_samples=30):
    """
    Crea una tabella dettagliata con predizioni e confidenze.
    
    Args:
        rejection_model: RejectionClassifier instance
        X_test: Test features
        y_test: True labels
        n_samples: Numero di campioni da mostrare
        
    Returns:
        DataFrame con tutte le informazioni richieste
    """
    # Limita ai primi n_samples
    X_subset = X_test[:n_samples]
    y_subset = y_test[:n_samples]
    
    # Ottieni predizioni con rejection
    predictions = rejection_model.predict(X_subset)
    
    # Ottieni probabilitÃ  per entrambe le classi
    probas = rejection_model.predict_proba(X_subset)
    
    # Ottieni i nomi delle classi dal modello base
    class_names = rejection_model.base_model.model.classes_
    
    # Crea lista di predicted labels con "I don't know" per i rifiutati
    predicted_labels = []
    
    for i in range(len(predictions)):
        if predictions[i] == rejection_model.rejection_label:
            predicted_labels.append("I don't know")
        else:
            predicted_labels.append(str(predictions[i]))
    
    # Crea DataFrame con colonne separate per ogni classe (arrotondate a 4 decimali)
    df = pd.DataFrame({
        'Sample_ID': range(1, n_samples + 1),
        'True_Label': y_subset.values,
        'Predicted_Label': predicted_labels,
        f'Prob_{class_names[0]}': probas[:, 0].round(4),  # ProbabilitÃ  prima classe (4 decimali)
        f'Prob_{class_names[1]}': probas[:, 1].round(4)   # ProbabilitÃ  seconda classe (4 decimali)
    })
    
    return df


def calculate_metrics(y_true, y_pred, rejection_label=-1):
    """
    Calcola le metriche richieste.
    
    Returns:
        Dictionary con accuracy, misclassification rate, rejection rate
    """
    n_total = len(y_pred)
    
    # Identifica campioni rifiutati e accettati
    rejected_mask = y_pred == rejection_label
    accepted_mask = ~rejected_mask
    
    n_rejected = np.sum(rejected_mask)
    n_accepted = np.sum(accepted_mask)
    
    # Calcola metriche solo su campioni accettati
    if n_accepted > 0:
        y_true_accepted = y_true[accepted_mask]
        y_pred_accepted = y_pred[accepted_mask]
        
        n_correct = np.sum(y_true_accepted == y_pred_accepted)
        n_wrong = n_accepted - n_correct
        
        accuracy = n_correct / n_accepted
        misclassification_rate = n_wrong / n_accepted
    else:
        accuracy = 0.0
        misclassification_rate = 0.0
    
    rejection_rate = n_rejected / n_total
    
    return {
        'accuracy': accuracy,
        'misclassification_rate': misclassification_rate,
        'rejection_rate': rejection_rate,
        'n_total': n_total,
        'n_accepted': n_accepted,
        'n_rejected': n_rejected,
        'n_correct': n_correct if n_accepted > 0 else 0,
        'n_wrong': n_wrong if n_accepted > 0 else 0
    }


def print_metrics_table(metrics):
    """Stampa una tabella formattata con le metriche"""
    print(f"\n{'='*80}")
    print(f"METRICHE DEL REJECTION CLASSIFIER")
    print(f"{'='*80}")
    print(f"\nCampioni totali: {metrics['n_total']}")
    print(f"  â€¢ Accettati: {metrics['n_accepted']} ({metrics['n_accepted']/metrics['n_total']*100:.2f}%)")
    print(f"    - Corretti: {metrics['n_correct']}")
    print(f"    - Sbagliati: {metrics['n_wrong']}")
    print(f"  â€¢ Rifiutati (I don't know): {metrics['n_rejected']} ({metrics['rejection_rate']*100:.2f}%)")
    
    print(f"\n{'â”€'*80}")
    print(f"{'Metrica':<40} {'Valore':>20} {'Percentuale':>15}")
    print(f"{'â”€'*80}")
    print(f"{'Accuracy (su campioni accettati)':<40} {metrics['accuracy']:>20.4f} {metrics['accuracy']*100:>14.2f}%")
    print(f"{'Misclassification Rate (su accettati)':<40} {metrics['misclassification_rate']:>20.4f} {metrics['misclassification_rate']*100:>14.2f}%")
    print(f"{'Rejection Rate (su totale)':<40} {metrics['rejection_rate']:>20.4f} {metrics['rejection_rate']*100:>14.2f}%")
    print(f"{'='*80}\n")


# ============================================================================
# PREPARAZIONE DATASET
# ============================================================================

print(f"\n{'='*80}")
print("PREPARAZIONE DATASET")
print(f"{'='*80}")

# Crea un'istanza del dataset
ds = dataset(
    file_path='/Users/matteopascuzzo/MachineLearningTesi/Datasets/HW_Failure/BackBlaze_2017_5PercRate_scikit.csv',
    dataset_name="Test Dataset",
    test_size=0.2,
    random_state=42,
    stratify=False
)

# Prepara il dataset
ds.preprocess()

# Ottieni i dati
X_train, X_test, y_train, y_test = ds.data

# Converti in numpy array se necessario
if hasattr(X_test, 'values'):
    X_test_array = X_test.values
else:
    X_test_array = X_test

if hasattr(y_test, 'values'):
    y_test_array = y_test.values
else:
    y_test_array = y_test


# ============================================================================
# DECISION TREE CON REJECTION OPTION
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING DECISION TREE CON REJECTION OPTION")
print(f"{'='*80}\n")

# Crea il Decision Tree base
dt_base = decision_tree(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    criterion='gini',
    random_state=42
)

# Addestra il modello
dt_base.train(X_train, y_train)

# Crea il Rejection Classifier con soglia 90%
dt_rejection = RejectionClassifier(
    base_model=dt_base,
    confidence_threshold=0.90
)

# ============================================================================
# GENERA TABELLA
# ============================================================================

print(f"\n{'='*80}")
print("TABELLA PREDIZIONI DETTAGLIATA (primi 100 campioni)")
print(f"{'='*80}\n")

# Crea la tabella
df_predictions = create_detailed_predictions_table(
    rejection_model=dt_rejection,
    X_test=X_test_array,
    y_test=y_test,
    n_samples=200
)

# Mostra la tabella a console
print(df_predictions.to_string(index=False))

print(f"\n{'â”€'*80}")
print("   Il file completo Ã¨ stato salvato come CSV!")
print(f"{'â”€'*80}")

# Salva in CSV
output_dir = Path('/Users/matteopascuzzo/Desktop')
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / 'predictions_with_IDontKnow.csv'

df_predictions.to_csv(csv_path, index=False, float_format='%.4f')

print(f"\n{'='*80}")
print(f"âœ“ TABELLA PREDIZIONI SALVATA!")
print(f"{'='*80}")
print(f"File: predictions_detailed.csv")
print(f"Path completo: {csv_path}")
print(f"Contenuto: {len(df_predictions)} campioni con predizioni dettagliate")
print(f"{'='*80}\n")


# ============================================================================
# CALCOLA E MOSTRA METRICHE
# ============================================================================

# Calcola metriche su tutto il test set
y_pred_full = dt_rejection.predict(X_test_array)
metrics = calculate_metrics(y_test_array, y_pred_full, rejection_label=-1)

# Stampa tabella delle metriche
print_metrics_table(metrics)


# ============================================================================
# ANALISI AGGIUNTIVA: DISTRIBUZIONE STATUS
# ============================================================================

print(f"{'='*80}")
print("DISTRIBUZIONE STATUS NEI PRIMI 30 CAMPIONI")
print(f"{'='*80}")
print("DISTRIBUZIONE PREDIZIONI NEI PRIMI 100 CAMPIONI")
print(f"{'='*80}\n")

# Conta le predizioni
n_rejected = (df_predictions['Predicted_Label'] == "I don't know").sum()
n_predicted = len(df_predictions) - n_rejected

# Conta corretti e sbagliati tra quelli predetti
n_correct = 0
n_wrong = 0
for idx, row in df_predictions.iterrows():
    if row['Predicted_Label'] != "I don't know":
        if str(row['Predicted_Label']) == str(row['True_Label']):
            n_correct += 1
        else:
            n_wrong += 1

print(f"{'Predizioni Corrette':<25} {n_correct:>3} campioni ({n_correct/len(df_predictions)*100:>5.1f}%)")
print(f"{'Predizioni Sbagliate':<25} {n_wrong:>3} campioni ({n_wrong/len(df_predictions)*100:>5.1f}%)")
print(f"{'Rejected (I don\'t know)':<25} {n_rejected:>3} campioni ({n_rejected/len(df_predictions)*100:>5.1f}%)")

print(f"\n{'='*80}\n")

print(f"\nðŸŽ¯ FILE CREATO:")
print(f"   ðŸ“Š predictions_detailed.csv â†’ Tabella con {len(df_predictions)} campioni")

# ============================================================================
# FUNZIONI PER DECISION TREE NORMALE (SENZA REJECTION)
# ============================================================================

def create_normal_predictions_table(dt_model, X_test, y_test, n_samples=30):
    """
    Crea una tabella dettagliata con predizioni e confidenze per DT normale.
    
    Args:
        dt_model: Decision Tree model (senza rejection)
        X_test: Test features
        y_test: True labels
        n_samples: Numero di campioni da mostrare
        
    Returns:
        DataFrame con tutte le informazioni richieste
    """
    # Limita ai primi n_samples
    X_subset = X_test[:n_samples]
    y_subset = y_test[:n_samples]
    
    # Ottieni predizioni normali (senza rejection)
    predictions = dt_model.predict(X_subset)
    
    # Ottieni probabilita per entrambe le classi
    probas = dt_model.model.predict_proba(X_subset)
    
    # Ottieni i nomi delle classi dal modello
    class_names = dt_model.model.classes_
    
    # Crea DataFrame con colonne separate per ogni classe (arrotondate a 4 decimali)
    df = pd.DataFrame({
        'Sample_ID': range(1, n_samples + 1),
        'True_Label': y_subset.values,
        'Predicted_Label': predictions,
        f'Prob_{class_names[0]}': probas[:, 0].round(4),
        f'Prob_{class_names[1]}': probas[:, 1].round(4)
    })
    
    return df


def calculate_normal_metrics(y_true, y_pred):
    """
    Calcola le metriche per un classificatore normale (senza rejection).
    
    Returns:
        Dictionary con accuracy e misclassification rate
    """
    n_total = len(y_pred)
    
    # Calcola metriche su tutti i campioni
    n_correct = np.sum(y_true == y_pred)
    n_wrong = n_total - n_correct
    
    accuracy = n_correct / n_total
    misclassification_rate = n_wrong / n_total
    
    return {
        'accuracy': accuracy,
        'misclassification_rate': misclassification_rate,
        'n_total': n_total,
        'n_correct': n_correct,
        'n_wrong': n_wrong
    }


def print_normal_metrics_table(metrics):
    """Stampa una tabella formattata con le metriche per DT normale"""
    print(f"\n{'='*80}")
    print(f"METRICHE DEL DECISION TREE NORMALE")
    print(f"{'='*80}")
    print(f"\nCampioni totali: {metrics['n_total']}")
    print(f"  * Corretti: {metrics['n_correct']} ({metrics['n_correct']/metrics['n_total']*100:.2f}%)")
    print(f"  * Sbagliati: {metrics['n_wrong']} ({metrics['n_wrong']/metrics['n_total']*100:.2f}%)")
    
    print(f"\n{'-'*80}")
    print(f"{'Metrica':<40} {'Valore':>20} {'Percentuale':>15}")
    print(f"{'-'*80}")
    print(f"{'Accuracy (su tutti i campioni)':<40} {metrics['accuracy']:>20.4f} {metrics['accuracy']*100:>14.2f}%")
    print(f"{'Misclassification Rate':<40} {metrics['misclassification_rate']:>20.4f} {metrics['misclassification_rate']*100:>14.2f}%")
    print(f"{'='*80}\n")


# ============================================================================
# DECISION TREE NORMALE (SENZA REJECTION OPTION)
# ============================================================================

print(f"\n\n{'#'*80}")
print(f"{'#'*80}")
print("DECISION TREE NORMALE (SENZA REJECTION OPTION)")
print(f"{'#'*80}")
print(f"{'#'*80}\n")

print(f"\n{'='*80}")
print("TRAINING DECISION TREE NORMALE")
print(f"{'='*80}\n")

# Crea un nuovo Decision Tree (stessi parametri del precedente)
dt_normal = decision_tree(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    criterion='gini',
    random_state=42
)

# Addestra il modello
dt_normal.train(X_train, y_train)

# ============================================================================
# GENERA TABELLA PER DT NORMALE
# ============================================================================

print(f"\n{'='*80}")
print("TABELLA PREDIZIONI DETTAGLIATA - DT NORMALE (primi 200 campioni)")
print(f"{'='*80}\n")

# Crea la tabella
df_predictions_normal = create_normal_predictions_table(
    dt_model=dt_normal,
    X_test=X_test_array,
    y_test=y_test,
    n_samples=200
)

# Mostra la tabella a console
print(df_predictions_normal.to_string(index=False))

print(f"\n{'-'*80}")
print("   Il file completo e stato salvato come CSV!")
print(f"{'-'*80}")

# Salva in CSV
csv_path_normal = output_dir / 'predictions_without_IDontKnow.csv'

df_predictions_normal.to_csv(csv_path_normal, index=False, float_format='%.4f')

print(f"\n{'='*80}")
print(f"OK TABELLA PREDIZIONI DT NORMALE SALVATA!")
print(f"{'='*80}")
print(f"File: predictions_without_IDontKnow.csv")
print(f"Path completo: {csv_path_normal}")
print(f"Contenuto: {len(df_predictions_normal)} campioni con predizioni dettagliate")
print(f"{'='*80}\n")


# ============================================================================
# CALCOLA E MOSTRA METRICHE PER DT NORMALE
# ============================================================================

# Calcola metriche su tutto il test set
y_pred_normal_full = dt_normal.predict(X_test_array)
metrics_normal = calculate_normal_metrics(y_test_array, y_pred_normal_full)

# Stampa tabella delle metriche
print_normal_metrics_table(metrics_normal)


# ============================================================================
# ANALISI AGGIUNTIVA: DISTRIBUZIONE STATUS PER DT NORMALE
# ============================================================================

print(f"{'='*80}")
print("DISTRIBUZIONE PREDIZIONI NEI PRIMI 200 CAMPIONI - DT NORMALE")
print(f"{'='*80}\n")

# Conta corretti e sbagliati
n_correct_normal = 0
n_wrong_normal = 0
for idx, row in df_predictions_normal.iterrows():
    if str(row['Predicted_Label']) == str(row['True_Label']):
        n_correct_normal += 1
    else:
        n_wrong_normal += 1

print(f"{'Predizioni Corrette':<25} {n_correct_normal:>3} campioni ({n_correct_normal/len(df_predictions_normal)*100:>5.1f}%)")
print(f"{'Predizioni Sbagliate':<25} {n_wrong_normal:>3} campioni ({n_wrong_normal/len(df_predictions_normal)*100:>5.1f}%)")

print(f"\n{'='*80}\n")

print(f"\nFILE CREATO:")
print(f"   predictions_without_IDontKnow.csv -> Tabella con {len(df_predictions_normal)} campioni")


# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

print(f"\n\n{'#'*80}")
print("RIEPILOGO FINALE")
print(f"{'#'*80}\n")

print("FILE CREATI:")
print(f"  1. predictions_with_IDontKnow.csv    - Decision Tree con Rejection Option")
print(f"  2. predictions_without_IDontKnow.csv - Decision Tree Normale\n")

print("METRICHE A CONFRONTO:")
print(f"{'-'*80}")
print(f"{'Metrica':<40} {'DT con Rejection':>20} {'DT Normale':>20}")
print(f"{'-'*80}")
print(f"{'Accuracy':<40} {metrics['accuracy']:>20.4f} {metrics_normal['accuracy']:>20.4f}")
print(f"{'Misclassification Rate':<40} {metrics['misclassification_rate']:>20.4f} {metrics_normal['misclassification_rate']:>20.4f}")
print(f"{'Rejection Rate':<40} {metrics['rejection_rate']:>20.4f} {'N/A':>20}")
print(f"{'-'*80}\n")