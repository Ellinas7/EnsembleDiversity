import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import qr
from scipy.stats import mode
from ML_algorithms.ML_algorithm import ML_algorithm


class random_rotation_forest(ML_algorithm):
    """Random Rotation Forest Classifier"""
    
    def __init__(self, n_estimators: int = 30, random_state: int = 42, 
                 n_jobs: int = -1, max_features=None, bootstrap=True, 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini"):
        super().__init__(n_estimators, random_state)
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.label_encoder_ = None
        self.estimators_ = []
        self.rotation_matrices_ = []
    
    def _random_rotation_matrix(self, n):
        """Genera una matrice di rotazione random ortogonale n x n"""
        r = np.random.normal(size=(n, n))
        Q, R = qr(r)
        M = np.dot(Q, np.diag(np.sign(np.diag(R))))
        if np.linalg.det(M) < 0:
            M[:, 0] = -M[:, 0]
        return M.astype(np.float32)
    
    def _create_model(self):
        return self
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train, y_train = check_X_y(X_train, y_train)
        
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        self.n_classes_ = len(self.classes_)
        
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        
        self.estimators_ = []
        self.rotation_matrices_ = []
        np.random.seed(self.random_state)
        
        print(f"Training {self.name}...")
        for i in range(self.n_estimators):
            tree_seed = None if self.random_state is None else self.random_state + i
            
            rotation_matrix = self._random_rotation_matrix(X_train.shape[1])
            self.rotation_matrices_.append(rotation_matrix)
            
            if self.bootstrap:
                n_samples = X_train.shape[0]
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X_train[indices]
                y_sample = y_encoded[indices]
            else:
                X_sample = X_train
                y_sample = y_encoded
            
            X_rotated = np.dot(X_sample, rotation_matrix)
            
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X_rotated, y_sample)
            self.estimators_.append(tree)
        
        self.model = self
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
