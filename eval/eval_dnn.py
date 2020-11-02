from sklearn.metrics import classification_report, accuracy_score
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout
from numpy.random import seed
from model.input_fn import load_evaluation_train_test_data


if __name__ == "__main__":
    set_random_seed(2)
    seed(1)

    # load input data
    X_train, Y_train, X_test, Y_test, _ = load_evaluation_train_test_data()
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()

    print X_train.shape
    print Y_train.shape
    print X_test.shape
    print Y_test.shape

    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=X_train.shape[1], kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='random_normal', activation='sigmoid'))

    # compile the Keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()

    # fit the keras model on the dataset
    model.fit(X_train, Y_train, epochs=20, batch_size=100)

    # evaluate with train example
    y_pred = model.predict_classes(X_train, batch_size=100, verbose=1)
    print "Evaluate with train dataset"
    print(classification_report(Y_train, y_pred))

    # evaluate with test example
    loss, accuracy = model.evaluate(X_test, Y_test)
    print "Evaluate with test dataset"
    print('Accuracy: %.2f' % (accuracy * 100))
    y_pred = model.predict_classes(X_test, batch_size=100, verbose=1)
    print(classification_report(Y_test, y_pred))

