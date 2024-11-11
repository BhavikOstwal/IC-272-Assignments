import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from A import load_and_preprocess_images

model = tf.keras.models.load_model('mymodel.h5')  

test_images, test_labels = load_and_preprocess_images("cifar-3class-data/test/", ["0", "1", "2"],False)

proba = model.predict(test_images)

pred_labels = np.argmax(proba, axis=1)
true_labels = np.argmax(test_labels, axis=1)

accuracy = accuracy_score(true_labels, pred_labels)

print(f"Test Accuracy: {accuracy * 100:.2f}%")