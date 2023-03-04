import os
import modules.util as util
import math
import numpy as np
import modules.datagenerator as dtgen
import modules.network as network
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

siamese_network.compile(loss= loss.loss(1), optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-3), # 0.5e-6
                    metrics=["accuracy"])

checkpoint_path = os.path.join(home, 'checkpoints')

# Loads the weights
siamese_network.load_weights(checkpoint_path)

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
history = siamese_network.fit(
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