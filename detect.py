from data_maker import load_data

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

from  sklearn.model_selection import train_test_split

(feature, labels)  = load_data()

x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size = 0.2)

categories = ["mask_weared_incorrect", "with_mask", "without_mask"]

model = tf.keras.models.load_model('mymodel.h5')



prediction = model.predict(x_test)
plt.figure(figsize=(9,9))
for i in  range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i])
    plt.xlabel('Actual:' + categories[y_test[i]]+'\n'+'Prediction: '+ categories[np.argmax(prediction[i])])

    plt.xticks([])

plt.show()

