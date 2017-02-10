from feature_extraction import extract_features_from_list, scale_features
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import glob
import numpy as np

if __name__ == "__main__":
    img_slice = slice(None)
    vehicle_images = glob.glob("./data/vehicles/**/*.png")
    non_vehicle_images = glob.glob("./data/non-vehicles/**/*.png")
    images = np.hstack((vehicle_images[img_slice], non_vehicle_images[img_slice]))

    features = extract_features_from_list(images)
    features = scale_features(features)
    num_vehicle_features = len(vehicle_images[img_slice])
    num_non_vehicle_features = len(non_vehicle_images[img_slice])

    labels = np.hstack((
        np.ones(num_vehicle_features),
        np.zeros(num_non_vehicle_features)))

    rand = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42)

    classifier = SVC(kernel="linear")

    print("Fitting data to classifier")
    classifier.fit(X_train, y_train)

    print("Getting classifier score")
    score = classifier.score(X_test, y_test)

    filename = "classifier.pkl"
    joblib.dump(classifier, filename)

    print("Score: {0:.4f}, saved to file: {1}".format(score, filename))
