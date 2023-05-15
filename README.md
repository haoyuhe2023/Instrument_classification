# Instrument_classification
Instrument recognition based on MFCC with CNN
This is a script for building and training a convolutional neural network (CNN) model on audio data for audio classification. The audio data is stored in a JSON file, which is loaded into the script using the load_data function. The data_prepare function is used to split the data into training, validation, and test sets, and to reshape the data for compatibility with the CNN architecture.

The CNN model is built using the model_build function, which defines the layers of the network. The model includes three convolutional layers, each followed by max pooling and batch normalization layers. The output from the convolutional layers is flattened and passed through a dense layer with ReLU activation and a dropout layer to reduce overfitting. The final output layer uses softmax activation to output a probability distribution over the 11 audio classes.

The model is trained using the fit method with the training and validation data, and the plot_res function is used to visualize the training and validation accuracy and error over the epochs. The model is evaluated using the test set and the evaluate method, and the confusion matrix is calculated using the confusion_matrix function from scikit-learn.
