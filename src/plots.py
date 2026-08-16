import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(history, metric):

    train_df = pd.DataFrame(history["train"])
    test_df = pd.DataFrame(history["test"])

    epochs = range(1, len(train_df) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_df[metric],
        label=f"Train {metric.capitalize()}"
    )

    plt.plot(
        epochs,
        test_df[metric],
        label=f"Test {metric.capitalize()}"
    )

    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Training vs Test {metric.capitalize()}")

    plt.legend()
    plt.grid()
    plt.show()