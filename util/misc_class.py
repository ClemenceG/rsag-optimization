import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd

class Metrics:
    """
    Abstract class to calculate metrics:
    - accuracy
    """
    def calc_accuracy(y_pred, labels):
        predicted_digits = y_pred.argmax(1)                            # pick digit with largest network output
        correct_ones = (predicted_digits == labels).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
        return correct_ones.sum().item()


class Visualizations:
    """
    Abstract class to plot losses and accuracies
    """
    def plot_losses(losses: pd.DataFrame, 
                    title='Losses', 
                    std_devs=None, 
                    save_path=None):

        if std_devs is not None:
            plt.errorbar(range(len(losses)), losses, yerr=std_devs)
        if len(losses.shape) >1:
            for i in range(losses.shape[1]):
                plt.plot(losses[:,i])
        else:
            plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        if save_path is not None:
            plt.savefig(save_path)

    def plot_accuracies(accuracies: list[float],
                        title='Accuracies',
                        std_devs=None,
                        save_path=None):
        plt.plot(accuracies)
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.show()