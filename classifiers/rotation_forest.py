from classifiers.classifier import Classifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode


class RotationForest(Classifier):
    """Rotation Forest Classifier (implementazione manuale)"""
    
    def __init__(self, n_estimators: int = 10, random_state: int = 42, 
                 n_jobs: int = 1, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, n_features_per_subset=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features_per_subset = n_features_per_subset
        self.label_encoder_ = None
        self.estimators_ = []
        self.rotation_matrices_ = []
    
    def _build_rotation_matrix(self, X, random_state):
        """Costruisce una matrice di rotazione usando PCA su sottoinsiemi di features"""
        n_features = X.shape[1]
        n_subsets = max(1, n_features // self.n_features_per_subset)
        
        rng = np.random.RandomState(random_state)
        feature_indices = rng.permutation(n_features)
        
        rotation_matrix = np.zeros((n_features, n_features), dtype=np.float32)
        
        start = 0
        for i in range(n_subsets):
            end = start + self.n_features_per_subset
            if i == n_subsets - 1:
                end = n_features
            
            subset_indices = feature_indices[start:end]
            X_subset = X[:, subset_indices]
            
            bootstrap_indices = rng.choice(X.shape[0], size=int(0.75 * X.shape[0]), replace=True)
            X_bootstrap = X_subset[bootstrap_indices]
            
            n_components = min(X_bootstrap.shape[0], X_bootstrap.shape[1], end - start)
            pca = PCA(n_components=n_components)
            pca.fit(X_bootstrap)
            
            for j, idx in enumerate(subset_indices):
                for k, comp_idx in enumerate(subset_indices):
                    if j < pca.components_.shape[0] and k < pca.components_.shape[1]:
                        rotation_matrix[idx, comp_idx] = pca.components_[j, k]
            
            start = end
        
        return rotation_matrix

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train, y_train = check_X_y(X_train, y_train)
        
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        self.n_classes_ = len(self.classes_)
        
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        
        self.estimators_ = []
        self.rotation_matrices_ = []
        
        print(f"Training {self.name}...")
        for i in range(self.n_estimators):
            tree_seed = None if self.random_state is None else self.random_state + i
            
            rotation_matrix = self._build_rotation_matrix(X_train, tree_seed)
            self.rotation_matrices_.append(rotation_matrix)
            
            X_rotated = np.dot(X_train, rotation_matrix)
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed
            )
            tree.fit(X_rotated, y_encoded)
            self.estimators_.append(tree)
        
        print(f"✓ {self.name} addestrato")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("Modello non ancora addestrato.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        X_test = check_array(X_test)
        
        predictions = []
        for tree, rot_matrix in zip(self.estimators_, self.rotation_matrices_):
            X_rotated = np.dot(X_test, rot_matrix)
            predictions.append(tree.predict(X_rotated))
        
        predictions = np.array(predictions)
        voted_predictions, _ = mode(predictions, axis=0, keepdims=False)
        
        return self.label_encoder_.inverse_transform(voted_predictions)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("Modello non ancora addestrato.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        X_test = check_array(X_test)
        
        all_proba = []
        for tree, rot_matrix in zip(self.estimators_, self.rotation_matrices_):
            X_rotated = np.dot(X_test, rot_matrix)
            all_proba.append(tree.predict_proba(X_rotated))
        
        return np.mean(all_proba, axis=0)
