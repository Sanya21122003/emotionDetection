# import libraries
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# Initialize image data generator with rescaling
trainDataGen=ImageDataGenerator(rescale=1./255)
validationDataGen=ImageDataGenerator(rescale=1./255)
# Preprocess all train images
trainGenerator=trainDataGen.flow_from_directory(
    'data/train',
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')
# Preprocess all test images
validationGenerator=validationDataGen.flow_from_directory(
    'data/test',
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')
# Create Model structure/CNN
emotionModel=Sequential()
emotionModel.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
emotionModel.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2,2)))
emotionModel.add(Dropout(0.25))

emotionModel.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2,2)))
emotionModel.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2,2)))
emotionModel.add(Dropout(0.25))

emotionModel.add(Flatten())
emotionModel.add(Dense(1024,activation='relu'))
emotionModel.add(Dropout(0.5))
emotionModel.add(Dense(7,activation='softmax'))

emotionModel.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001,decay=1e-6),metrics=['accuracy']) 
# Train the Model
emotionModelInfo=emotionModel.fit_generator(
    trainGenerator,
    steps_per_epoch=28709//64,
    epochs=50,
    validation_data=validationGenerator,
    validation_steps=7178//64)
# Save model structure
modelJson=emotionModel.to_json()
with open("emotionModel.json","w") as json_file:
  json_file.write(modelJson)
# Save trained model weight in .h5 file
emotionModel.save_weights('emotionModel.h5')