import numpy as np

NR_LABELS = 12

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
    # Here we have homework-1 instead of homework-2 due to an off-by-one error
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

    for i in range(50):
        train_model(X_tr, y_train, W, b, 0.01)


    # see how many it got correct on training data
    test_scores = np.dot(X_tr, np.transpose(W)) + b
    Y_test_prediction = np.argmax(test_scores, axis = 1)


    count_correct = int(np.sum(Y_test_prediction == y_train))
    print(count_correct)
