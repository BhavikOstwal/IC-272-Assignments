import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from A import load_and_preprocess_images
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_images, train_labels = load_and_preprocess_images("cifar-3class-data/train/", ["0", "1", "2"],False)


print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)

train_images, val_images, train_labels_one_hot, val_labels_one_hot = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
print(train_images.shape)
print(train_labels_one_hot.shape)


def my_cnn():
    model = Sequential([
        tf.keras.Input(shape=(32, 32, 3)),
        # Input layer: Image shape (32, 32, 3) is implicitly inferred
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(100, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model


cnn_model = my_cnn()


history = cnn_model.fit(train_images, train_labels_one_hot, epochs=50, batch_size=200, validation_data=(val_images, val_labels_one_hot))

cnn_model.save('mymodel.h5')


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Epoch-wise Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
