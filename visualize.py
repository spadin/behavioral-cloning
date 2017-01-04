from model_architecture import model_architecture
from keras.utils.visualize_util import plot

def visualize():
    model = model_architecture()
    plot(model, to_file="model.png", show_shapes=True)

if __name__ == "__main__":
    visualize()
