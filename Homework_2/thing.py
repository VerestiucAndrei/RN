import numpy as np
import pandas as pd

NR_LABELS = 12
LR_DECAY = 0.05
LEARNING_RATE = 0.01

def softmax(array):
    shifted = array - np.max(array)

    exp_array = np.exp(shifted)
    return_array = exp_array / np.sum(exp_array)

    return return_array



def train_model(train_set, train_labels, W, b, learning_rate):

    indices = np.random.permutation(len(train_set))

    for i in indices:
        sample = train_set[i]
        sample_label = train_labels[i]

        prediction = np.dot(W, sample) + b

        probs = softmax(prediction)

        error = probs.copy()
        error[sample_label] -= 1.0

        dW = np.outer(error, sample)
        db = error

        W -= learning_rate * dW
        b -= learning_rate * db



if __name__ == "__main__":
    train = np.load("train.npz")
    test = np.load("test.npz")

    N, Dim = train["data"].shape

    W = np.zeros((NR_LABELS, Dim))
    b = np.zeros(NR_LABELS)


    X_train = train["data"]
    y_train = train["labels"]
    X_test = test["data"]

    X_tr = X_train / 255.0
    X_te = X_test / 255.0

    # Split into training and validation data
    X_training, X_validation = X_tr[:48000], X_tr[48000:]
    Y_training, Y_validation = y_train[:48000], y_train[48000:]

    for epoch in range(50):
        learning_rate = LEARNING_RATE / (1.0 + LR_DECAY * epoch)
        train_model(X_training, Y_training, W, b, learning_rate)


    # See how many it got correct on training data
    test_scores = np.dot(X_validation, np.transpose(W)) + b
    Y_test_prediction = np.argmax(test_scores, axis = 1)


    count_correct = int(np.sum(Y_test_prediction == Y_validation))
    print(count_correct)


    # Submission
    test_scores = np.dot(X_te, np.transpose(W)) + b
    predictions = np.argmax(test_scores, axis = 1)

    submission = pd.DataFrame({
        "ID": range(len(predictions)),
        "target": predictions,
    })

    submission.to_csv("submission.csv", index = False)
    print("Created submission.csv")
