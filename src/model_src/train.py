import joblib
import warnings
import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description='Train Iris Classifier with MLflow tracking')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Maximum number of iterations (default: 200)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--solver', type=str, default='lbfgs',
                        choices=['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                        help='Algorithm to use in optimization (default: lbfgs)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size ratio (default: 0.2)')
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None,
                        help='MLflow tracking server URI. If not set, MLflow tracks locally.')
    parser.add_argument('--no-register-model', action='store_true',
                        help='Do not register the model in MLflow Model Registry.')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    warnings.filterwarnings('ignore')
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    else:
        print("MLflow tracking URI not set, tracking locally to './mlruns'")

    # Load and split data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.random_state
    )

    # Create an input example
    input_example = X_train[0:1]

    MODEL_NAME = "iris_classifier_model"

    # Set experiment name
    mlflow.set_experiment("iris_experiment_1")

    # Start MLflow run
    with mlflow.start_run(run_name="logistic_regression"):
        # Define and train model with parameters from arguments
        params = {
            "max_iter": args.max_iter,
            "random_state": args.random_state,
            "solver": args.solver
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_params(params)
        
        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1
        })
        
        # Log and optionally register model
        if not args.no_register_model:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=MODEL_NAME
            )
            
            # Get the client and latest version
            client = mlflow.tracking.MlflowClient()
            latest_version = max([
                int(mv.version) for mv in 
                client.search_model_versions(f"name='{MODEL_NAME}'")
            ])
            
            print(f"Model {MODEL_NAME} version {latest_version} has been registered")
        else:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example
            )
            print("Model logged as an artifact but not registered.")

        # Save model locally
        joblib.dump(model, "iris_model.pkl")
        
        print(f"Training completed with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()