from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from numpy.random import seed
from tensorflow import set_random_seed
from model.input_fn import load_evaluation_train_test_data


if __name__ == "__main__":
    # set random seed
    set_random_seed(2)
    seed(1)

    # load input datax
    print "Preparing training and testing dataset"
    X_train, Y_train, X_test, Y_test, _ = load_evaluation_train_test_data()

    print "Start to train XGBoost"
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()

    print X_train.shape
    print Y_train.shape
    print X_test.shape
    print Y_test.shape

    # Train model
    model = XGBClassifier()
    model.fit(X_train, Y_train)

    # make prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)

    print "Evaluate with test dataset"
    print('Accuracy: %.2f' % (accuracy * 100))
    print(classification_report(Y_test, y_pred))