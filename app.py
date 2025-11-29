import streamlit as st

from experiment_logger import ExperimentLogger

# Algoritmi
from ML_algorithms.random_forest import random_forest
from ML_algorithms.extra_trees import extra_trees
from ML_algorithms.adaboost import adaboost
from ML_algorithms.xgboost import xgboost
from ML_algorithms.light_gbm import light_gbm
from ML_algorithms.catboost import catboost
from ML_algorithms.gradient_boosting_decision_trees import gradient_boosting_decision_trees
from ML_algorithms.random_patches import random_patches
from ML_algorithms.rotation_forest import rotation_forest
from ML_algorithms.random_rotation_forest import random_rotation_forest

# Configurazione pagina
st.set_page_config(page_title="Experiment Runner", layout="wide")
st.title("🧪 Dual Rejection Experiment Runner")

# Dizionario algoritmi
ALGORITHMS = {
    "Random Forest": random_forest,
    "Extra Trees": extra_trees,
    "AdaBoost": adaboost,
    "XGBoost": xgboost,
    "LightGBM": light_gbm,
    "CatBoost": catboost,
    "Gradient Boosting": gradient_boosting_decision_trees,
    "Random Patches": random_patches,
    "Rotation Forest": rotation_forest,
    "Random Rotation Forest": random_rotation_forest
}

# Inizializza logger per accedere ai dataset
logger = ExperimentLogger()

# Sidebar
st.sidebar.header("⚙️ Configurazione")

# 1. Dataset
st.sidebar.subheader("1. Dataset")
dataset_name = st.sidebar.selectbox("Seleziona dataset", list(logger.DATASETS.keys()))

# 2. Algoritmo
st.sidebar.subheader("2. Algoritmo")
algo_name = st.sidebar.selectbox("Seleziona algoritmo", list(ALGORITHMS.keys()))
n_estimators = st.sidebar.number_input("N. estimators", min_value=2, max_value=100, value=2)

# 3. Parametri Rejection
st.sidebar.subheader("3. Rejection Parameters")
static_threshold = st.sidebar.slider("Static threshold", 0.5, 0.99, 0.9)
rejection_percentile = st.sidebar.slider("Rejection percentile", 1.0, 50.0, 10.0)

# 4. Gruppi Metriche
st.sidebar.subheader("4. Gruppi Metriche")
use_classiche = st.sidebar.checkbox("Classiche", value=True)
use_doppio = st.sidebar.checkbox("Doppio (Voting 2/2 + Recovery)", value=True)
use_terna = st.sidebar.checkbox("Terna (Majority + Single Vote)", value=False)

# Costruisci lista gruppi
metric_groups = []
if use_classiche:
    metric_groups.append('classiche')
if use_doppio:
    metric_groups.append('doppio')
if use_terna:
    metric_groups.append('terna')

# Info pannello principale
st.subheader("📋 Configurazione Esperimento")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Dataset:** {dataset_name}")
    st.write(f"**Algoritmo:** {algo_name}")
    st.write(f"**N. estimators:** {n_estimators}")
with col2:
    st.write(f"**Static threshold:** {static_threshold}")
    st.write(f"**Rejection percentile:** {rejection_percentile}%")
    st.write(f"**Gruppi metriche:** {metric_groups}")

# Bottone esegui
if st.sidebar.button("🚀 Esegui Esperimento", type="primary"):
    if not metric_groups:
        st.error("❌ Seleziona almeno un gruppo di metriche!")
    else:
        with st.spinner("Esecuzione in corso..."):
            try:
                # Crea algoritmo
                algo = ALGORITHMS[algo_name](n_estimators=n_estimators)
                
                # Esegui dual rejection
                results_static, results_percentile = logger.run_dual_rejection(
                    dataset_name,
                    algo,
                    metric_groups,
                    static_threshold=static_threshold,
                    rejection_percentile=rejection_percentile
                )
                
                st.success("✅ Esperimenti salvati nel Megafile.csv!")
                
                # Mostra riassunto
                st.subheader("📊 Risultati")
                
                st.write(f"**Esperimento #{results_static['experiment_number']}:** {results_static['experiment_name']}")
                st.write(f"**Esperimento #{results_percentile['experiment_number']}:** {results_percentile['experiment_name']}")
                
            except Exception as e:
                st.error(f"❌ Errore: {str(e)}")

# Sezione cancellazione esperimenti
st.sidebar.markdown("---")
st.sidebar.subheader("🗑️ Gestione Esperimenti")

if len(logger.df) > 0:
    # Mostra lista esperimenti
    exp_options = {f"#{row['experiment_number']} - {row['experiment_name']}": row['experiment_number'] 
                   for _, row in logger.df.iterrows()}
    
    selected_exp = st.sidebar.selectbox("Seleziona esperimento da cancellare", list(exp_options.keys()))
    
    if st.sidebar.button("🗑️ Cancella Esperimento", type="secondary"):
        exp_num = exp_options[selected_exp]
        logger.delete_experiment(exp_num)
        st.sidebar.success(f"✅ Esperimento #{exp_num} cancellato!")
        st.rerun()
else:
    st.sidebar.write("Nessun esperimento presente.")