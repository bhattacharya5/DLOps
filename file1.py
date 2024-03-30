import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the random values
randomValues = np.array([-3.5, -1.2, 0, 3.2, -3.8, 1.7, -0.6, 3.3, -2.6, 5.6])


# Model Configuration consists of three convolutional layers with increasing filter sizes.
# Batch normalization layers are added after each convolutional layer to normalize the activations.
# Global average pooling is used instead of flattening before the final dense layer.
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])


# Print model summary
model.summary()

plt.tight_layout()
plt.show()

