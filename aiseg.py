import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Add more layers if necessary...

    up1 = UpSampling2D(size=(2, 2))(conv2)
    conv8 = Conv2D(64, 2, activation='relu', padding='same')(up1)
    merge1 = concatenate([conv1, conv8], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model

# Assuming you have loaded your dataset into X_train and Y_train
model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, batch_size=2, epochs=50)

predicted_mask = model.predict(some_new_image)

import matplotlib.pyplot as plt

plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')
plt.title('Predicted Segmentation Mask')
plt.show()
