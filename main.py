from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Paths to train and validation directories
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Image dimensions and batch size
img_size = (224, 224)
batch_size = 32

# Data augmentation for training data
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,      # Increase rotation range
    width_shift_range=0.15,  # Increase width shift
    height_shift_range=0.15, # Increase height shift
    shear_range=0.3,        # Add more shear
    zoom_range=0.3,         # Increase zoom range
    horizontal_flip=True,   # Flip images horizontally
    brightness_range=[0.8, 1.2]  # Vary brightness
)

# Only rescaling for validation data
val_gen = ImageDataGenerator(rescale=1.0 / 255)

# Load training data
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load validation data
val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze the first 100 layers
    layer.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)  # Increase units for better learning
x = Dropout(0.2)(x)                   # Increase dropout to avoid overfitting
x = Dense(128, activation='relu')(x)  # Add another dense layer
x = Dropout(0.2)(x)
output = Dense(3, activation='softmax')(x)  # 3 classes: CRPF, BSF, JKP

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with class weights
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,  # Increase epochs for better learning
    steps_per_epoch=train_data.samples // batch_size,
    validation_steps=val_data.samples // batch_size,
    class_weight=class_weights
)

# Save the trained model
model.save('soldier_uniform_classifier.h5')

# Evaluate the model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Classification report and confusion matrix
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(val_data.classes, y_pred_classes))

print("Classification Report:")
print(classification_report(val_data.classes, y_pred_classes))


