
import matplotlib.pyplot as plt


def plot_history(history):
    """
    From Keras documentation https://keras.io/visualization/#training-history-visualization
    :param history:
    :return:
    """
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()