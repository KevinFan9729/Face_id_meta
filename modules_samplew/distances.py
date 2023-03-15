import tensorflow as tf
from tensorflow.keras import layers

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