from training.common import train_disease


if __name__ == "__main__":
    result = train_disease("heart_disease")
    print(f"Best model: {result['best_model_validation_rank']}")
