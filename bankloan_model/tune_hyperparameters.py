import optuna
import json
from bankloan_model.train_pipeline import objective, load_dataset
from bankloan_model.config.core import config
from sklearn.model_selection import train_test_split

def main():
    # Load training data
    data = load_dataset(file_name=config.app_config_.training_data_file)

    # Divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],
        data[config.model_config_.loan_status],
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state,
    )

    # Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=10)

    # Best trial
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.params}")

    # Save best trial to JSON file
    with open("best_trial.json", "w") as f:
        json.dump(best_trial.params, f)

if __name__ == "__main__":
    main()