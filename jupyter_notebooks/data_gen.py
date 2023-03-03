import keras
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

#if there is issue with cuda, disable cuda
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


random.seed(15)#fixed random state 
np.random.seed(15)


home=os.path.abspath(os.getcwd())
# data_path=os.path.join(home, '101_ObjectCategories')
# data_path=os.path.join(home, '256_moded')
data_path=os.path.join(home, 'data')

#global data
pairs=[]
classes=[]

for file in os.listdir(data_path):
    classes.append(file)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        x1 = np.empty((self.batch_size, 224,224,3))
        x2 = np.empty((self.batch_size, 224,224,3))
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # print(self.labels[int(ID)])
            path1, path2, label, s1, s2 = self.labels[int(ID)]

            _x1 = cv2.imread(path1)
            _x2 = cv2.imread(path2)
            _x1 = preprocess(_x1, s1)
            _x1 = scale_back(_x1) / 255.
            _x2 = preprocess(_x2, s2)
            _x2 = scale_back(_x2) / 255.

            x1[i,] = _x1
            x2[i,] = _x2
            y[i,] = label
        return [x1, x2], y



def make_pairs():#makes pairs of data
    global pairs, classes, labels
    # pairs = np.array(pairs).astype("float32")
    # labels = np.array(labels).astype("float32")
    # pairs = []
    for class_ in classes:
        class_path = os.path.join(data_path, class_)
        for img_path in os.listdir(class_path):
            if np.random.uniform()<=0.25:#rescale images
                image1 = os.path.join(class_path, img_path)
                image_select=random.choice(os.listdir(class_path))
                image2 = os.path.join(class_path, image_select)
                scale = np.random.uniform(0.3,0.6)#scaling factor
                select_index = random.choice([1,2])
                if select_index==1:
                    s1=int(scale*224)#scale down
                    s2 = 224
                    scale_flag=1
                else:
                    s2=int(scale*224)#scale down
                    s1 = 224
                    scale_flag=2
                pairs+=[[image1, image2, 1, s1, s2]]

                class_select = random.choice(classes)
                while class_select == class_:# keep trying if select the current class
                    class_select = random.choice(classes)
                class_path2 = os.path.join(data_path, class_select)
                image_select=random.choice(os.listdir(class_path2))
                image2 = os.path.join(class_path2, image_select)
                if scale_flag ==1:
                    s1 = 224
                    if np.random.uniform()<0.5:
                        s2=int(scale*224)#scale down
                    else:
                        s2 = 224
                elif scale_flag ==2:
                    if np.random.uniform()<0.5:
                        select_index = random.choice([1,2])
                        if select_index==1:
                            s1=int(scale*224)#scale down
                            s2 = 224
                        else:
                            s2=int(scale*224)#scale down
                            s1 = 224
                scale_flag=0
                pairs+=[[image1, image2, 0, s1, s2]]

            image1 = os.path.join(class_path, img_path)
            image_select=random.choice(os.listdir(class_path))
            image2 = os.path.join(class_path, image_select)
            # image1=preprocess(image1)
            # image2=preprocess(image2)
            pairs+=[[image1, image2, 1, 224, 224]]


            class_select = random.choice(classes)
            while class_select == class_:# keep trying if select the current class
                class_select = random.choice(classes)
            class_path2 = os.path.join(data_path, class_select)
            image_select=random.choice(os.listdir(class_path2))
            image2 = os.path.join(class_path2, image_select)
            # image2=preprocess(image2)
            pairs+=[[image1, image2, 0, 224, 224]]




def preprocess(img, size=224, interpolation =cv2.INTER_AREA):
    #extract image size
    h, w = img.shape[:2]
    #check color channels
    c = None if len(img.shape) < 3 else img.shape[2]
    #square images have an aspect ratio of 1:1
    if h == w: 
        return cv2.resize(img, (size, size), interpolation)
    elif h>w:#height is larger
        diff= h-w
        img=cv2.copyMakeBorder(img,0,0,int(diff/2.0),int(diff/2.0),cv2.BORDER_CONSTANT, value = 0)
        # img=cv2.copyMakeBorder(img,0,0,int(diff/2.0),int(diff/2.0),cv2.BORDER_REPLICATE)
    elif h<w:
        diff= w-h
        # img=cv2.copyMakeBorder(img,int(diff/2.0),int(diff/2.0),0,0,cv2.BORDER_REPLICATE)
        img=cv2.copyMakeBorder(img,int(diff/2.0),int(diff/2.0),0,0,cv2.BORDER_CONSTANT, value = 0)
    return cv2.resize(img, (size, size), interpolation)

def scale_back(img, size =224):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  dif = size-h
  x_pos = int((dif)/2.0)
  y_pos = int((dif)/2.0)
  mask = np.zeros((size, size, c), dtype=img.dtype)
  mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
#   print(mask.shape)
  # cv2.imshow("test",mask)
  # cv2.waitKey(0)
  return mask

make_pairs()


#60/20/20
partition = {'train': np.arange(len(pairs)*.6),
             'validation': np.arange(len(pairs)*.6,len(pairs)*.8),
             'test': np.arange(len(pairs)*.8, len(pairs))}

# Generators
train_generator = DataGenerator(partition['train'], pairs, batch_size=16)
val_generator = DataGenerator(partition['validation'], pairs, batch_size=16)

#2*16*224*224*3
#first index: which batch
#2nd index: image or label
#3rd index: which image of the pair (x1 or x2)
#4th index: which element of the batch
img = train_generator[0][0][1][0]
label = train_generator[0][1]
plt.imshow(img)


def make_embedding():
    # first block
    input_=layers.Input((224, 224, 3))
    x = layers.BatchNormalization()(input_)
    x = layers.Conv2D(64, (3,3), activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
    
    # second block
    x = layers.Conv2D(128, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
    
    x = layers.Conv2D(256, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)


    x = layers.Conv2D(512, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
    

    x = layers.Conv2D(512, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
    
    # final block
    #kernel l2 1e-4
    # x = layers.Conv2D(32, (1,1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(1e-2))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(2048, activation='relu',kernel_regularizer=regularizers.l2(1e-2))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='sigmoid',kernel_regularizer=regularizers.l2(1e-3))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)#maybe remove this?
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(512, activation='softmax')(x)
    # x = layers.Dense(256, activation='sigmoid')(x)
    # x = layers.Dense(256, activation='sigmoid')(x)
    
    return keras.Model(inputs=input_, outputs=x, name="embedding")


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        return tf.math.abs(embedding1-embedding2)


class L1Dist_mod(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        return tf.math.reduce_sum(tf.math.abs(embedding1-embedding2), axis=1, keepdims=True)

class L2Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        sum_square = tf.math.reduce_sum(tf.math.square(embedding1 - embedding2), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

class cosine(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        return 1-tf.keras.losses.cosine_similarity(embedding1,embedding2)

class TF_L2Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        return tf.math.reduce_euclidean_norm(embedding1,embedding2)



def make_siamese_net():
    input1=layers.Input((224, 224, 3))
    input2=layers.Input((224, 224, 3))
    embedding=make_embedding()#same embedding on both heads

    
    siamese_layer=L1Dist_mod()
    # # cosine_similiarity=tf.keras.layers.Dot(axes=-1,normalize=True)([embedding_left_output,embedding_right_output])
    distances=siamese_layer(embedding(input1),embedding(input2))
    distances = layers.BatchNormalization()(distances)#maybe remove this?
    # # distances = layers.Dropout(0.5)(distances)
    classifier = layers.Dense(1, activation='sigmoid')(distances)

    # cosine_similiarity=tf.keras.layers.Dot(axes=-1,normalize=True)([embedding(input1),embedding(input2)])

    #We are using acos so that highly similiar sentences can have well 
    #seperation. So we are not using:
    #cos_distance=1-cosine_similiarity
    #https://math.stackexchange.com/questions/2874940/cosine-similarity-vs-angular-distance?newreg=02fd1e16a9164cbba05197b28d353409
    # clip_cosine_similarities = tf.clip_by_value(cosine_similiarity, -1.0, 1.0)
    # import math as m
    # pi = tf.constant(m.pi,dtype= tf.float32)
    # cos_distance = 1.0 - (tf.acos(clip_cosine_similarities)/pi)
    #Acos Range (0 to Pi (3.14)/pi radians, with 0 as closest 1 as farthest )
    #cos_distance range=1-0=>1 to 1-1=>0, with 1 being nearest and 0 being farthest
    #http://mathonweb.com/help_ebook/html/functions_2.htm
    return keras.Model(inputs=[input1,input2],outputs=classifier,name="siamese_network")
    # return keras.Model(inputs=[input1,input2],outputs=classifier,name="siamese_network")
siamese_net=make_siamese_net()

# siamese_net.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
#                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-4), 
#                     metrics=["accuracy"])
siamese_net.compile(loss=loss(1), 
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-6), 
                    metrics=["accuracy"])
# checkpoint_path = os.path.join(home, 'checkpoints',"weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
checkpoint_path = os.path.join(home, 'checkpoints')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', 
                             verbose=1, save_best_only=True, mode='max')
early_callback=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=35, restore_best_weights=True)

# callbacks_list = [early_callback,checkpoint]
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        lr = lr * tf.math.exp(-0.05)
        if lr >= 1e-8:
            return lr
        else:
            return 1e-8
# 0.5e-7
class print_lr(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(keras.backend.eval(self.model.optimizer.lr))

lr_sched = LearningRateScheduler(scheduler)



callbacks_list = [early_callback, lr_sched, checkpoint, print_lr()]



# Fit the model
# siamese_net = keras.models.load_model('checkpoints', custom_objects={ 'contrastive_loss': loss(1) })

batch_size=32
epochs=500
history = siamese_net.fit(
    train_generator,
    validation_data=val_generator,
    callbacks=callbacks_list,
    epochs=epochs,
)
# history = siamese_net.fit(
#     [x_train_1, x_train_2],
#     y_train,
#     validation_data=([x_val_1, x_val_2], y_val),
#     batch_size=batch_size,
#     callbacks=callbacks_list,
#     epochs=epochs,
#     # sample_weight=sample_weight,
# )

print(max(history.history["val_accuracy"]))
