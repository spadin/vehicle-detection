from feature_extraction import extract_features_from_list, scale_training_features
from model import save_model
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import glob
import numpy as np

def train(img_slice=slice(None)):
    vehicle_images = glob.glob("./data/vehicles/**/*.png")
    non_vehicle_images = glob.glob("./data/non-vehicles/**/*.png")
    images = np.hstack((vehicle_images[img_slice], non_vehicle_images[img_slice]))

    print("Extracting features")
    features = extract_features_from_list(images, filetype="png")

    print("Scaling features")
    features, scaler = scale_training_features(features)
    num_vehicle_features = len(vehicle_images[img_slice])
    num_non_vehicle_features = len(non_vehicle_images[img_slice])

    labels = np.hstack((
        np.ones(num_vehicle_features),
        np.zeros(num_non_vehicle_features)))

    rand = np.random.randint(0, 100)

    print("Splitting features")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=rand)

    classifier = LinearSVC()

    print("Fitting data to classifier")
    classifier.fit(X_train, y_train)

    print("Getting classifier score")
    score = classifier.score(X_test, y_test)

    print("Score: {0:.4f}".format(score))

    return classifier, scaler

if __name__ == "__main__":
    classifier, scaler = train()
    save_model(classifier, scaler)
