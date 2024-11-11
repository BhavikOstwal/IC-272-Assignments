import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_preprocess_images(base_path, folders, forA = True):
    images = []
    labels = []

    for lb, folder in enumerate(folders):
        folder_path = os.path.join(base_path, folder)
        for filename in os.listdir(folder_path):
            img = Image.open(os.path.join(folder_path, filename))
            if (forA):
                img = img.resize((32, 32)).convert("L")
                img_array = np.array(img, dtype=np.float32).flatten()
            else:
                img_array = np.array(img, dtype=np.float32)

            img_array /= 255.0 

            images.append(img_array)
            labels.append(lb)  

    images = np.array(images)
    labels = np.array(labels)

    labels = tf.keras.utils.to_categorical(labels, num_classes=len(np.unique(labels)))
    return images, labels


if __name__ == "__main__":


    def visualize_images(images, labels, num_images=4):
        # plt.figure(figsize=(8, 8))
        
        for i in range(num_images):
            img = images[i].reshape(32, 32)
            label = labels[i]

            plt.subplot(2, 2, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Label: {label}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()


    train_images, train_labels = load_and_preprocess_images("cifar-3class-data/train/" , ["0", "1", "2"])
    test_images, test_labels = load_and_preprocess_images("cifar-3class-data/test/", ["0", "1", "2"])

    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    # print("Test images shape:", test_images.shape)
    # print("Test labels shape:", test_labels.shape)

    visualize_images(train_images, train_labels, num_images=4)

    train_images, val_images, train_labels_one_hot, val_labels_one_hot = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
    # print(train_images.shape)
    # print(train_labels_one_hot.shape)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1024,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    mymodel = model.fit(train_images, train_labels_one_hot, epochs=500, batch_size=200, validation_data=(val_images, val_labels_one_hot), verbose=1)

    plt.plot(mymodel.history['accuracy'], label='Training Accuracy')
    plt.plot(mymodel.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig("A-I.png")
    plt.show()

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print(test_loss)
    print((test_acc*100).__format__(".2f"))