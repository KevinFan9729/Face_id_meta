import os
import modules.util as util
import math
import numpy as np
import modules.datagenerator as dtgen
import modules.network_PoolerCLS as network
import modules.loss as loss
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from tensorflow import keras

home=os.path.abspath(os.getcwd())
data_path=os.path.join(home, 'data_train_valset')

#global data
pairs=[]
classes=[]

for file in os.listdir(data_path):
    classes.append(file)

util.make_pairs(data_path, pairs, classes)

# partition = {'train': np.arange(math.floor(len(pairs)*.6)),
#              'validation': np.arange(math.floor(len(pairs)*.6),math.floor(len(pairs)*.8)),
#              'test': np.arange(math.floor(len(pairs)*.8), math.floor(len(pairs)))}

partition = {'train': np.arange(math.floor(len(pairs)*.75)),
             'validation': np.arange(math.floor(len(pairs)*.75),math.floor(len(pairs)))}

print(f"Number of Train Pairs: {len(partition['train'])}")
print(f"Number of Validation Pairs: {len(partition['validation'])}")

# Generators
train_generator = dtgen.DataGenerator(partition['train'], pairs, batch_size=16)
val_generator = dtgen.DataGenerator(partition['validation'], pairs, batch_size=16)


siamese_obj = network.SiameseNetwork()
siamese_network = siamese_obj.make_siamese_net()

siamese_network.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.5e-3)
siamese_network.compile(loss= loss.loss(1), optimizer=opt, # 0.5e-6 # ViT paper lr=8e-4
                    metrics=["accuracy"])

checkpoint_path = os.path.join(home, 'checkpoint')

# Loads the weights
# siamese_network.load_weights(checkpoint_path)

# checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', 
                            #  verbose=1, save_best_only=True, mode='max')
epochs=500
# sample_weights = np.ones(math.floor(len(pairs))
lr = 0.5e-3
for epoch in range(0,epochs):
    if epoch > 15:
        lr *= tf.math.exp(-0.001)
        opt.learning_rate.assign(lr)
    
    history = siamese_network.fit_generator(train_generator, validation_data=val_generator, epochs=1)
    shuffled_ytrue = []
    if epoch % 10 == 1:
        pred = siamese_network.predict(train_generator)
        for index in train_generator.indexes:
            shuffled_ytrue.append(train_generator.labels[index][2])
        for i in range(0,len(shuffled_ytrue)):
            if shuffled_ytrue[i] == pred[i]:
                train_generator.labels[train_generator.indexes[i]][-1] = 1
            else:
                train_generator.labels[train_generator.indexes[i]][-1] = 1.25

    if (epoch == 0):
        print(type(history))
        print(history)
        print(history.history.keys())
        print(f'Validation list: {history.history["val_accuracy"]}')
        best_val = history.history["val_accuracy"][-1]
        siamese_network.save("checkpoint")
    elif (history.history["val_accuracy"][-1] > best_val):
        best_val = history.history["val_accuracy"][-1]
        siamese_network.save("checkpoint")

print(max(history.history["val_accuracy"]))