import torch
import numpy as np
import xgboost as xgb
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader
from dataset import SpectrogramDataset
from config import (
    DATA_PATH, CLASSIFIER_BATCH_SIZE, LEARNING_RATE, SEED, 
    MODELS_PATH, RESULTS_PATH, SAMPLING_RATE, 
    CHIMPANZEE_DATA_PATH, CLASS_WEIGHTS
)
from scipy.stats import randint, uniform

# Set the random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

def extract_features(dataset, max_samples=None, feature_reduction='flatten', pca_components=100):
    """
    Extract features and labels from SpectrogramDataset with optional dimensionality reduction
    
    Parameters:
    -----------
    dataset : SpectrogramDataset
        The spectrogram dataset to extract features from
    max_samples : int, optional
        Maximum number of samples to process (for memory-constrained environments)
    feature_reduction : str, optional
        Method for feature reduction: 'flatten', 'mean_std', 'pca'
    pca_components : int, optional
        Number of components to keep when using PCA
        
    Returns:
    --------
    tuple: (features, labels)
        features: numpy array of extracted features
        labels: numpy array of class labels
    """
    features = []
    labels = []
    
    # Create a DataLoader to iterate through the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    sample_count = 0
    for sample, label in dataloader:
        # Stop if we've reached max_samples
        if max_samples and sample_count >= max_samples:
            break
            
        # Extract log spectrogram from the sample
        spectrogram = sample['data'].squeeze().numpy()
        
        # Apply feature extraction/reduction based on method
        if feature_reduction == 'flatten':
            # Simple flattening (can lead to high-dimensional features)
            extracted_features = spectrogram.flatten()
        
        elif feature_reduction == 'mean_std':
            # Extract statistical features (mean, std, min, max for each frequency band)
            means = np.mean(spectrogram, axis=1)
            stds = np.std(spectrogram, axis=1)
            mins = np.min(spectrogram, axis=1)
            maxs = np.max(spectrogram, axis=1)
            # You can add more statistical features here
            extracted_features = np.concatenate([means, stds, mins, maxs])
            
        elif feature_reduction == 'pca':
            # Flatten first, PCA will be applied later in the pipeline
            extracted_features = spectrogram.flatten()
            
        else:
            raise ValueError(f"Unknown feature reduction method: {feature_reduction}")
        
        features.append(extracted_features)
        labels.append(label.item())
        sample_count += 1
    
    return np.array(features), np.array(labels)

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate a trained model with comprehensive metrics
    
    Parameters:
    -----------
    model : estimator
        Trained machine learning model
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    class_names : list
        List of class names
    
    Returns:
    --------
    dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities for ROC-AUC if the model supports it
    try:
        y_prob = model.predict_proba(X_test)
        has_probabilities = True
    except (AttributeError, NotImplementedError):
        has_probabilities = False
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Prepare results dictionary
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
        'classification_report': classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True
        )
    }
    
    # Add ROC-AUC and Precision-Recall AUC metrics if probabilities are available
    if has_probabilities and len(class_names) == 2:  # Binary classification
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        results['roc_auc'] = roc_auc
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
        pr_auc = auc(recall, precision)
        results['pr_auc'] = pr_auc
    
    elif has_probabilities and len(class_names) > 2:  # Multi-class
        # One-vs-Rest ROC-AUC
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        results['roc_auc'] = roc_auc
    
    return results

def plot_learning_curves(model_name, param_values, scores, param_name, score_name='F1 Score'):
    """
    Plot learning curves showing how a parameter affects model performance
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    param_values : list
        List of parameter values tested
    scores : list
        List of corresponding scores
    param_name : str
        Name of the parameter being varied
    score_name : str, optional
        Name of the score metric
    """
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, scores, marker='o')
    plt.title(f'Effect of {param_name} on {model_name} Performance')
    plt.xlabel(param_name)
    plt.ylabel(score_name)
    plt.grid(True)
    
    # Save the plot
    os.makedirs(f'{RESULTS_PATH}/plots', exist_ok=True)
    plt.savefig(f'{RESULTS_PATH}/plots/{model_name}_{param_name}_curve.png')
    plt.close()

def perform_hyperparameter_tuning(
    model_class, 
    param_distributions, 
    dataset_path, 
    val_dataset_path=None, 
    duration=2, 
    target_sample_rate=SAMPLING_RATE, 
    random_state=SEED, 
    scoring='f1_weighted',
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    feature_reduction='flatten',
    max_samples=None,
    class_weight=None
):
    """
    Perform hyperparameter tuning using RandomizedSearchCV with pipelines
    
    Parameters:
    -----------
    model_class : class
        The machine learning model class to tune
    param_distributions : dict
        Dictionary of hyperparameter distributions to sample from
    dataset_path : str
        Path to the training dataset
    val_dataset_path : str, optional
        Path to the validation dataset
    duration : int, optional
        Duration of audio clips
    target_sample_rate : int, optional
        Target sampling rate for audio processing
    random_state : int, optional
        Controls the shuffling applied to the data before split
    scoring : str or callable, optional
        Scoring metric for model evaluation
    n_iter : int, optional
        Number of parameter settings sampled in RandomizedSearchCV
    cv : int or cross-validation generator, optional
        Cross-validation strategy
    feature_reduction : str, optional
        Method for feature reduction
    max_samples : int, optional
        Maximum number of samples to process
    class_weight : dict or 'balanced', optional
        Class weights for handling imbalanced datasets
    
    Returns:
    --------
    dict: A dictionary containing tuning results and best model
    """
    start_time = time.time()
    
    # Create SpectrogramDataset for training
    print(f"Loading training dataset from {dataset_path}...")
    train_ds = SpectrogramDataset(dataset_path, duration=duration, target_sample_rate=target_sample_rate)
    
    # Extract features with the specified reduction method
    print(f"Extracting features using {feature_reduction} method...")
    
    if val_dataset_path:
        # Separate train and validation datasets
        test_dataset = SpectrogramDataset(val_dataset_path, duration=duration, target_sample_rate=target_sample_rate)
        X_train, y_train = extract_features(train_ds, max_samples=max_samples, feature_reduction=feature_reduction)
        X_test, y_test = extract_features(test_dataset, max_samples=max_samples, feature_reduction=feature_reduction)
    else:
        # Split training dataset for validation
        X, y = extract_features(train_ds, max_samples=max_samples, feature_reduction=feature_reduction)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    
    # Check for class imbalance
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution in training set: {dict(zip(unique_labels, counts))}")
    
    class_distribution = np.bincount(y_train)
    if np.max(class_distribution) / np.min(class_distribution) > 3:
        print("Warning: Dataset is imbalanced. Consider using class_weight='balanced'")
        
    # Get model name for logging
    model_name = model_class.__name__
    
    # Add model-specific pipeline steps
    steps = []
    
    # Always add a scaler
    steps.append(('scaler', StandardScaler()))
    
    # For high-dimensional features, consider feature selection
    if feature_reduction == 'flatten' and X_train.shape[1] > 1000:
        steps.append(('feature_selection', SelectKBest(f_classif, k=min(1000, X_train.shape[1] // 10))))
    
    # Adjust parameters for model instantiation
    model_params = {}
    
    # Add class_weight to supported models
    if class_weight is not None:
        if model_class in [RandomForestClassifier, SVC, LogisticRegression]:
            model_params['class_weight'] = class_weight
        elif model_class == xgb.XGBClassifier and class_weight == 'balanced':
            # Calculate scale_pos_weight for binary classification
            if len(unique_labels) == 2:
                neg_count = counts[0]
                pos_count = counts[1]
                model_params['scale_pos_weight'] = neg_count / pos_count
    
    # Add random_state to models that support it
    if hasattr(model_class(), 'random_state'):
        model_params['random_state'] = random_state
        
    # Add the model to the pipeline
    steps.append(('classifier', model_class(**model_params)))
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    # Adjust parameter names for the pipeline
    pipeline_param_distributions = {'classifier__' + key: value for key, value in param_distributions.items()}
    
    # Add preprocessor parameters if needed
    if 'feature_selection' in dict(steps):
        pipeline_param_distributions['feature_selection__k'] = randint(100, min(2000, X_train.shape[1]))
    
    # Create a stratified k-fold for more balanced evaluation
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Perform RandomizedSearchCV
    print(f"Starting hyperparameter tuning for {model_name} with {n_iter} iterations...")
    random_search = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=pipeline_param_distributions, 
        n_iter=n_iter,
        scoring=scoring, 
        cv=cv_strategy, 
        n_jobs=-1,  # Use all available cores
        verbose=2,
        random_state=random_state,
        return_train_score=True  # Useful for learning curves
    )
    
    # Fit the random search
    try:
        random_search.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model fitting: {str(e)}")
        return {
            'error': str(e),
            'model_name': model_name,
            'success': False
        }
    
    # Best model prediction and evaluation
    best_model = random_search.best_estimator_
    
    # Save model learning curves for key parameters
    cv_results = random_search.cv_results_
    for param_name in param_distributions.keys():
        pipeline_param_name = f'classifier__{param_name}'
        if pipeline_param_name in cv_results['params'][0]:
            # Extract parameter values and corresponding scores
            param_values = []
            scores = []
            
            for i, params in enumerate(cv_results['params']):
                if pipeline_param_name in params:
                    param_values.append(params[pipeline_param_name])
                    scores.append(cv_results['mean_test_score'][i])
            
            if param_values and scores:
                sorted_indices = np.argsort(param_values)
                sorted_values = [param_values[i] for i in sorted_indices]
                sorted_scores = [scores[i] for i in sorted_indices]
                
                # Plot learning curve for this parameter
                plot_learning_curves(model_name, sorted_values, sorted_scores, param_name)
    
    # Evaluate on test set
    evaluation_results = evaluate_model(best_model, X_test, y_test, train_ds.classes)
    
    # Combine with tuning results
    results = {
        'best_model': best_model,
        'best_params': {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()},
        'classes': train_ds.classes,
        'feature_reduction': feature_reduction,
        'n_features': X_train.shape[1],
        'elapsed_time': time.time() - start_time,
        'model_name': model_name,
        'success': True
    }
    
    # Merge with evaluation results
    results.update(evaluation_results)
    
    # Print detailed evaluation results
    print(f"\n{model_name} Results:")
    print("Best Hyperparameters:", results['best_params'])
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        best_model.predict(X_test), 
        target_names=train_ds.classes, 
        zero_division=0
    ))
    
    # Print overall metrics
    print(f"\nNumber of classes: {len(train_ds.classes)}")
    print(f"Classes: {train_ds.classes}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1 Score: {results['f1_score']:.4f}")
    
    if 'roc_auc' in results:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    return results

def save_model_and_results(results, base_filename):
    """
    Save model, hyperparameters, and evaluation results
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results and best hyperparameters
    base_filename : str
        Base filename for saving results
    """
    if not results.get('success', False):
        print(f"Skipping save for unsuccessful model: {results.get('model_name', 'unknown')}")
        return
        
    # Create directory if it doesn't exist
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Get model name
    model_name = results['model_name']
    
    # Convert numpy types to native Python types for JSON
    json_safe_results = {}
    for k, v in results.items():
        if k == 'best_model':
            continue  # Skip the model object
        elif isinstance(v, np.integer):
            json_safe_results[k] = int(v)
        elif isinstance(v, np.floating):
            json_safe_results[k] = float(v)
        elif isinstance(v, np.ndarray):
            json_safe_results[k] = v.tolist()
        else:
            json_safe_results[k] = v
    
    # Save model
    model_path = f"{MODELS_PATH}/{base_filename}.joblib"
    joblib.dump(results['best_model'], model_path)
    print(f"Model saved to {model_path}")
    
    # Save results
    results_path = f"{RESULTS_PATH}/{base_filename}.json"
    with open(results_path, 'w') as f:
        json.dump(json_safe_results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    # Create confusion matrix visualization
    if 'confusion_matrix' in results:
        plt.figure(figsize=(10, 8))
        cm = np.array(results['confusion_matrix'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        
        classes = results['classes']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        os.makedirs(f'{RESULTS_PATH}/plots', exist_ok=True)
        plt.savefig(f'{RESULTS_PATH}/plots/{base_filename}_confusion_matrix.png')
        plt.close()

def create_results_summary(results_list):
    """
    Create a summary DataFrame of model results
    
    Parameters:
    -----------
    results_list : list
        List of model results dictionaries
    
    Returns:
    --------
    pandas.DataFrame: Summary of model results
    """
    # Filter out failed models
    successful_results = [r for r in results_list if r.get('success', False)]
    
    if not successful_results:
        print("No successful models to summarize")
        return None
    
    summary_data = {
        'model': [],
        'accuracy': [],
        'f1_score': [],
        'feature_reduction': [],
        'n_features': [],
        'elapsed_time': [],
        'best_params': []
    }
    
    # Add ROC AUC if available
    has_roc_auc = all('roc_auc' in r for r in successful_results)
    if has_roc_auc:
        summary_data['roc_auc'] = []
    
    for result in successful_results:
        summary_data['model'].append(result['model_name'])
        summary_data['accuracy'].append(result['accuracy'])
        summary_data['f1_score'].append(result['f1_score'])
        summary_data['feature_reduction'].append(result.get('feature_reduction', 'N/A'))
        summary_data['n_features'].append(result.get('n_features', 'N/A'))
        summary_data['elapsed_time'].append(result.get('elapsed_time', 'N/A'))
        summary_data['best_params'].append(str(result['best_params']))
        
        if has_roc_auc:
            summary_data['roc_auc'].append(result['roc_auc'])
    
    df = pd.DataFrame(summary_data)
    
    # Sort by F1 score descending
    df = df.sort_values('f1_score', ascending=False)
    
    return df

if __name__=="__main__":
    # Define hyperparameter distributions for RandomizedSearchCV
    rf_param_distributions = {
        'n_estimators': randint(50, 500),
        'max_depth': [None] + list(randint(5, 50).rvs(5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.9]
    }
    
    xgb_param_distributions = {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 12),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'min_child_weight': randint(1, 10)
    }
    
    svm_param_distributions = {
        'C': uniform(0.1, 100),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(3)),
        'degree': randint(2, 5)  # For poly kernel
    }
    
    lr_param_distributions = {
        'C': uniform(0.1, 100),
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'l1_ratio': uniform(0, 1),  # For elasticnet
        'max_iter': randint(100, 1000)
    }
    
    # Feature reduction methods to try
    feature_reduction_methods = ['flatten', 'mean_std']
    
    # List to store results
    all_results = []
    
    # Try different feature reduction methods
    for feature_method in feature_reduction_methods:
        print(f"\n=== Using feature reduction method: {feature_method} ===\n")
        
        # Hyperparameter tuning for Random Forest
        print("\n=== Tuning Random Forest Hyperparameters ===")
        rf_base_filename = f"random_forest_{feature_method}"
        rf_tuning_results = perform_hyperparameter_tuning(
            RandomForestClassifier, 
            rf_param_distributions, 
            f"{CHIMPANZEE_DATA_PATH}/train",
            f"{CHIMPANZEE_DATA_PATH}/val",
            feature_reduction=feature_method,
            class_weight='balanced',
            n_iter=30  # Adjust based on your computational resources
        )
        if rf_tuning_results.get('success', False):
            save_model_and_results(rf_tuning_results, rf_base_filename)
            all_results.append(rf_tuning_results)
        
        # Hyperparameter tuning for XGBoost
        print("\n=== Tuning XGBoost Hyperparameters ===")
        xgb_base_filename = f"xgboost_{feature_method}"
        xgb_tuning_results = perform_hyperparameter_tuning(
            xgb.XGBClassifier, 
            xgb_param_distributions, 
            f"{CHIMPANZEE_DATA_PATH}/train",
            f"{CHIMPANZEE_DATA_PATH}/val",
            feature_reduction=feature_method,
            class_weight='balanced',
            n_iter=30
        )
        if xgb_tuning_results.get('success', False):
            save_model_and_results(xgb_tuning_results, xgb_base_filename)
            all_results.append(xgb_tuning_results)
        
        # Hyperparameter tuning for SVM
        print("\n=== Tuning SVM Hyperparameters ===")
        svm_base_filename = f"svm_{feature_method}"
        svm_tuning_results = perform_hyperparameter_tuning(
            SVC, 
            svm_param_distributions, 
            f"{CHIMPANZEE_DATA_PATH}/train",
            f"{CHIMPANZEE_DATA_PATH}/val",
            feature_reduction=feature_method,
            class_weight='balanced',
            n_iter=20  # SVM can be slower to train
        )
        if svm_tuning_results.get('success', False):
            save_model_and_results(svm_tuning_results, svm_base_filename)
            all_results.append(svm_tuning_results)
        
        # Hyperparameter tuning for Logistic Regression (fast and often effective)
        print("\n=== Tuning Logistic Regression Hyperparameters ===")
        lr_base_filename = f"logistic_regression_{feature_method}"
        lr_tuning_results = perform_hyperparameter_tuning(
            LogisticRegression, 
            lr_param_distributions, 
            f"{CHIMPANZEE_DATA_PATH}/train",
            f"{CHIMPANZEE_DATA_PATH}/val",
            feature_reduction=feature_method,
            class_weight='balanced',
            n_iter=20
        )
        if lr_tuning_results.get('success', False):
            save_model_and_results(lr_tuning_results, lr_base_filename)
            all_results.append(lr_tuning_results)
    
    # Create a summary CSV of results across all methods
    results_summary = create_results_summary(all_results)
    if results_summary is not None:
        results_summary.to_csv(f'{RESULTS_PATH}/model_comparison_summary.csv', index=False)
        
        # Create bar chart of model performances
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        models = results_summary['model'] + ' (' + results_summary['feature_reduction'] + ')'
        f1_scores = results_summary['f1_score']
        accuracies = results_summary['accuracy']
        
        # Sort by F1 score
        sorted_indices = np.argsort(f1_scores)[::-1]  # Descending order
        models = [models.iloc[i] for i in sorted_indices]
        f1_scores = [f1_scores.iloc[i] for i in sorted_indices]
        accuracies = [accuracies.iloc[i] for i in sorted_indices]
        
        # Plot
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width/2, f1_scores, width, label='F1 Score')
        ax.bar(x + width/2, accuracies, width, label='Accuracy')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_PATH}/plots/model_comparison.png')
        plt.close()
        
        # Print summary for quick reference
        print("\n=== Model Comparison Summary ===")
        print(results_summary)
        print(f"\nResults saved to {RESULTS_PATH}")
    else:
        print("No successful models to summarize")