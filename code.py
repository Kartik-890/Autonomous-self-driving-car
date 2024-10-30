1. !git clone https://github.com/rslim087a/track

1. !ls track

1. !pip3 install imgaug

1. import os
2. import numpy as np
3. import matplotlib.pyplot as plt
4. import matplotlib.image as mpimg
5. import keras
6. from keras.models import Sequential
7. from keras.optimizers import Adam
8. from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
9. from sklearn.utils import shuffle
10. from sklearn.model_selection import train_test_split
11. from imgaug import augmenters as iaa
12. import cv2
13. import pandas as pd
14. import ntpath
15. import random

1. datadir = 'track'
2. columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
3. data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
4. pd.set_option('display.max_colwidth', -1)
5. data.head()

1. def path_leaf(path):
2.   head, tail = ntpath.split(path)
3.   return tail
4. data['center'] = data['center'].apply(path_leaf)
5. data['left'] = data['left'].apply(path_leaf)
6. data['right'] = data['right'].apply(path_leaf)
7. data.head()

1. num_bins = 25
2. samples_per_bin = 400
3. hist, bins = np.histogram(data['steering'], num_bins)
4. center = (bins[:-1]+ bins[1:]) * 0.5
5. plt.bar(center, hist, width=0.05)
6. plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

1. print('total data:', len(data))
2. remove_list = []
3. for j in range(num_bins):
4.   list_ = []
5.   for i in range(len(data['steering'])):
6.     if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
7.       list_.append(i)
8.   list_ = shuffle(list_)
9.   list_ = list_[samples_per_bin:]
10.   remove_list.extend(list_)
11.  
12. print('removed:', len(remove_list))
13. data.drop(data.index[remove_list], inplace=True)
14. print('remaining:', len(data))
15.  
16. hist, _ = np.histogram(data['steering'], (num_bins))
17. plt.bar(center, hist, width=0.05)
18. plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))


1. print(data.iloc[1])
2. def load_img_steering(datadir, df):
3.   image_path = []
4.   steering = []
5.   for i in range(len(data)):
6.     indexed_data = data.iloc[i]
7.     center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
8.     image_path.append(os.path.join(datadir, center.strip()))
9.     steering.append(float(indexed_data[3]))
10.     # left image append
11.     image_path.append(os.path.join(datadir,left.strip()))
12.     steering.append(float(indexed_data[3])+0.15)
13.     # right image append
14.     image_path.append(os.path.join(datadir,right.strip()))
15.     steering.append(float(indexed_data[3])-0.15)
16.   image_paths = np.asarray(image_path)
17.   steerings = np.asarray(steering)
18.   return image_paths, steerings
19.  
20. image_paths, steerings = load_img_steering(datadir + '/IMG', data)

1. X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
2. print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

1. fig, axes = plt.subplots(1, 2, figsize=(12, 4))
2. axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
3. axes[0].set_title('Training set')
4. axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
5. axes[1].set_title('Validation set')


1. def zoom(image):
2.   zoom = iaa.Affine(scale=(1, 1.3))
3.   image = zoom.augment_image(image)
4.   return image

1. image = image_paths[random.randint(0, 1000)]
2. original_image = mpimg.imread(image)
3. zoomed_image = zoom(original_image)
4.  
5. fig, axs = plt.subplots(1, 2, figsize=(15, 10))
6. fig.tight_layout()
7.  
8. axs[0].imshow(original_image)
9. axs[0].set_title('Original Image')
10.  
11. axs[1].imshow(zoomed_image)
12. axs[1].set_title('Zoomed Image')


1. def pan(image):
2.   pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
3.   image = pan.augment_image(image)
4.   return image

1. image = image_paths[random.randint(0, 1000)]
2. original_image = mpimg.imread(image)
3. panned_image = pan(original_image)
4.  
5. fig, axs = plt.subplots(1, 2, figsize=(15, 10))
6. fig.tight_layout()
7.  
8. axs[0].imshow(original_image)
9. axs[0].set_title('Original Image')
10.  
11. axs[1].imshow(panned_image)
12. axs[1].set_title('Panned Image')

1. def img_random_brightness(image):
2.     brightness = iaa.Multiply((0.2, 1.2))
3.     image = brightness.augment_image(image)
4.     return image

1. image = image_paths[random.randint(0, 1000)]
2. original_image = mpimg.imread(image)
3. brightness_altered_image = img_random_brightness(original_image)
4.  
5. fig, axs = plt.subplots(1, 2, figsize=(15, 10))
6. fig.tight_layout()
7.  
8. axs[0].imshow(original_image)
9. axs[0].set_title('Original Image')
10.  
11. axs[1].imshow(brightness_altered_image)
12. axs[1].set_title('Brightness altered image ')


1. def img_random_flip(image, steering_angle):
2.     image = cv2.flip(image,1)
3.     steering_angle = -steering_angle
4.     return image, steering_angle

1. random_index = random.randint(0, 1000)
2. image = image_paths[random_index]
3. steering_angle = steerings[random_index]
4.  
5.  
6. original_image = mpimg.imread(image)
7. flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)
8.  
9. fig, axs = plt.subplots(1, 2, figsize=(15, 10))
10. fig.tight_layout()
11.  
12. axs[0].imshow(original_image)
13. axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
14.  
15. axs[1].imshow(flipped_image)
16. axs[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering_angle))

1. def random_augment(image, steering_angle):
2.     image = mpimg.imread(image)
3.     if np.random.rand() < 0.5:
4.       image = pan(image)
5.     if np.random.rand() < 0.5:
6.       image = zoom(image)
7.     if np.random.rand() < 0.5:
8.       image = img_random_brightness(image)
9.     if np.random.rand() < 0.5:
10.       image, steering_angle = img_random_flip(image, steering_angle)
11.     
12.     return image, steering_angle

1. ncol = 2
2. nrow = 10
3.  
4. fig, axs = plt.subplots(nrow, ncol, figsize=(15, 50))
5. fig.tight_layout()
6.  
7. for i in range(10):
8.   randnum = random.randint(0, len(image_paths) - 1)
9.   random_image = image_paths[randnum]
10.   random_steering = steerings[randnum]
11.     
12.   original_image = mpimg.imread(random_image)
13.   augmented_image, steering = random_augment(random_image, random_steering)
14.     
15.   axs[i][0].imshow(original_image)
16.   axs[i][0].set_title("Original Image")
17.   
18.   axs[i][1].imshow(augmented_image)
19.   axs[i][1].set_title("Augmented Image")
20.  

1. def img_preprocess(img):
2.     img = img[60:135,:,:]
3.     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
4.     img = cv2.GaussianBlur(img,  (3, 3), 0)
5.     img = cv2.resize(img, (200, 66))
6.     img = img/255
7.     return img

1. image = image_paths[100]
2. original_image = mpimg.imread(image)
3. preprocessed_image = img_preprocess(original_image)
4.  
5. fig, axs = plt.subplots(1, 2, figsize=(15, 10))
6. fig.tight_layout()
7. axs[0].imshow(original_image)
8. axs[0].set_title('Original Image')
9. axs[1].imshow(preprocessed_image)
10. axs[1].set_title('Preprocessed Image')

1. def batch_generator(image_paths, steering_ang, batch_size, istraining):
2.   
3.   while True:
4.     batch_img = []
5.     batch_steering = []
6.     
7.     for i in range(batch_size):
8.       random_index = random.randint(0, len(image_paths) - 1)
9.       
10.       if istraining:
11.         im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
12.      
13.       else:
14.         im = mpimg.imread(image_paths[random_index])
15.         steering = steering_ang[random_index]
16.       
17.       im = img_preprocess(im)
18.       batch_img.append(im)
19.       batch_steering.append(steering)
20.     yield (np.asarray(batch_img), np.asarray(batch_steering))  

1. x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
2. x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
3.  
4. fig, axs = plt.subplots(1, 2, figsize=(15, 10))
5. fig.tight_layout()
6.  
7. axs[0].imshow(x_train_gen[0])
8. axs[0].set_title('Training Image')
9.  
10. axs[1].imshow(x_valid_gen[0])
11. axs[1].set_title('Validation Image')

1. def nvidia_model():
2.   model = Sequential()
3.   model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
4.   model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
5.   model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
6.   model.add(Convolution2D(64, 3, 3, activation='elu'))
7.   
8.   model.add(Convolution2D(64, 3, 3, activation='elu'))
9. #   model.add(Dropout(0.5))
10.   
11.   
12.   model.add(Flatten())
13.   
14.   model.add(Dense(100, activation = 'elu'))
15. #   model.add(Dropout(0.5))
16.   
17.   model.add(Dense(50, activation = 'elu'))
18. #   model.add(Dropout(0.5))
19.   
20.   model.add(Dense(10, activation = 'elu'))
21. #   model.add(Dropout(0.5))
22.  
23.   model.add(Dense(1))
24.   
25.   optimizer = Adam(lr=1e-3)
26.   model.compile(loss='mse', optimizer=optimizer)
27.   return model

1. model = nvidia_model()
2. print(model.summary())

1. history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
2.                                   steps_per_epoch=300, 
3.                                   epochs=10,
4.                                   validation_data=batch_generator(X_valid, y_valid, 100, 0),
5.                                   validation_steps=200,
6.                                   verbose=1,
7.                                   shuffle = 1)

1. plt.plot(history.history['loss'])
2. plt.plot(history.history['val_loss'])
3. plt.legend(['training', 'validation'])
4. plt.title('Loss')
5. plt.xlabel('Epoch')

1. model.save('model.h5')

1. from google.colab import files
files.download('model.h5')
