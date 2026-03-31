''' 
Project 2
Model: Convolutional Neural Network (CNN) for Image Classification
Fresh Fruit VS Rotten fruit classification
Name: Jalyin Gonzalez 
Date: 3/29/2026
'''

import os
import numpy as np
import pandas as pd
import seaborn as sns

#Import used specifically for matplotlib.use("Agg")
import matplotlib
import matplotlib.pyplot as plt

#Tensorflow and CNN imports
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

''' 
Important to take note the subparts of each step as they are labeled in the code. However they may not be in order
due to the nature of the code and needing intialization for certain parts for use in the future.
'''

'''
Step 0 --- Initial Setup ---
a. Save figures
b. Configuration
c. Data folder
Inital setup before main preprocessing section
Function to save figures (confusion matrix, loss and accuracy curves, etc.)
Tried to use show() for simplicity of code but it was not working so I had to use savefig() and "Agg" approach instead.
'''
def save_figure(filename):
    matplotlib.use("Agg") 
    
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("Saved figure:", path)
    plt.close()
    
#Directory to save figures for outputs 
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

#configuration => sets image size to 224X224 pixels and batch size to 32
DATASET_ROOT = os.environ.get("DATASET_PATH", "Fruit Freshness Dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

print("Using data folder:", os.path.abspath(DATASET_ROOT))


'''
Step 1 --- Data Processing ---
 a. Load and preprocess the dataset
'''
#class names and labels for the dataset => automatically crawls through the dataset to assign lables (0 for Fresh and 1 for Rotten)
class_names = ["Fresh", "Rotten"]
label_to_int = {"Fresh": 0, "Rotten": 1}
all_paths, all_labels = [], []
#Loops for the automatic crawl
for fruit in sorted(os.listdir(DATASET_ROOT)):
    fruit_dir = os.path.join(DATASET_ROOT, fruit)
    if not os.path.isdir(fruit_dir):
        continue
    for state in class_names:
        state_dir = os.path.join(fruit_dir, state)
        if not os.path.isdir(state_dir):
            continue
        for f in os.listdir(state_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                all_paths.append(os.path.join(state_dir, f))
                all_labels.append(label_to_int[state])

if len(all_paths) < 2 or len(set(all_labels)) < 2:
    raise ValueError('Expected images under "<fruit>/Fresh" and "<fruit>/Rotten".')

# c. Split into training and validation(testing) sets
#30% of the data is used for testing and 70% for training. However I have tried a 20/80 split as well.
train_paths, val_paths, train_y, val_y = train_test_split(
    all_paths,
    all_labels,
    test_size=0.3,
    random_state=SEED,
    stratify=all_labels,
)
#Prints the total number of images, training images, and testing images to better understand the dataset and the split
print("Total images:", len(all_paths))
print("Training images:", len(train_paths))
print("Testing images:", len(val_paths))


# b. Resize images to 224x224 pixels and normalize pixel values 
sample_img_path = all_paths[0]

if sample_img_path:
    img = image.load_img(sample_img_path)
    img_array = image.img_to_array(img)
    print("Min pixel value:", np.min(img_array))
    print("Max pixel value:", np.max(img_array))

    normalized_img = img_array / 255.0
    #visualize the before and after normalization for checking the normalization process
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array.astype("uint8"))
    plt.title("Before Normalization")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(normalized_img)
    plt.title("After Normalization")
    plt.axis("off")
    save_figure("01_before_after_normalization.png")


#RGB image parsing function => reads the image, decodes it, resizes it to 224x224 pixels, and normalizes the pixel values as well as converts bytes into 3e array of num
#channels of 3 for RGB images
def parse_image(path, label):
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


#tensorflow processing pipeline => randomizes for epoches 
train_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(train_paths), np.array(train_y, dtype=np.int32))
)
#important to note the shuffle and reshuffle each iteration for the training set to prevent overfitting and to make sure the model doesnt learn patterns based on the order of the files 
train_ds = train_ds.shuffle(len(train_paths), seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(parse_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(val_paths), np.array(val_y, dtype=np.int32))
)
#groups the images into batches of 32 and prefetches the next batch for faster processing
val_ds = val_ds.map(parse_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

num_classes = len(class_names)

'''Step 2 --- Data Visualization ---
a. Display sample images from the dataset'''
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(min(9, images.shape[0])):
        plt.subplot(3, 3, i + 1)
        plt.imshow(np.clip(images[i].numpy(), 0, 1))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.suptitle("Sample training images", fontsize=14)
plt.tight_layout()
save_figure("02_sample_training_images.png")

#b. show class distribution in the training set
#prints the class distribution in the training set
classes_list = class_names
counts = [sum(1 for y in all_labels if y == i) for i in range(len(class_names))]
print(pd.DataFrame({"Class": classes_list, "Number of Images": counts}))
#visualizes the class distribution in the training set
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(classes_list, counts)
ax.bar_label(bars, labels=[str(int(c)) for c in counts], fontsize=12, fontweight="bold", padding=4)
ax.set_title("Class Distribution (Fresh vs Rotten — counts on bars)")
ax.set_xlabel("Classes")
ax.set_ylabel("Number of Images")
save_figure("03_class_distribution.png")

'''Step 3 --- Model Building/design ---
a. CNN Layers'''
#Sequential model => a linear stack of layers
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))

#b. pooling layers
#pooling layers => reduces the spatial dimensions of the feature maps by taking the maximum value of the feature map
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D((2, 2)))

#c. fully connected (Dense)layers
#flattens the feature maps into a 1D vector for the fully connected layers
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.summary()

'''Step 4 --- Model Training ---
a. Compile the model with appropriate loss function, optimizer, and metrics
'''
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

#b . Train the model on the training set and validate on the validation set
#15 epoches is a good balance between training time and model performance
EPOCHS = 15
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

'''Step 5 --- Model Evaluation ---
a. accuracy, precision, recall, F1-score, confusion matrix, and loss curves'''
#predicts the labels for the validation set and compares them to the true labels
y_true = []
y_pred = []
for batch_images, batch_labels in val_ds:
    preds = model.predict(batch_images, verbose=0)
    y_true.extend(batch_labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Validation accuracy:", np.mean(y_true == y_pred))
print(
    "\nClassification report:\n",
    classification_report(y_true, y_pred, target_names=class_names),
)
#confusion matrix => a table that shows the number of true and false positives and negatives
cm = confusion_matrix(y_true, y_pred)
#visualizes the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
save_figure("04_confusion_matrix.png")

#visualizes the loss and accuracy curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.tight_layout()
save_figure("05_loss_and_accuracy.png")
