
import tensorflow as tf
class tf_backend:

    def __init__(self,vocal=True):
        if vocal == True:

            self.meth = [attr for attr in dir(self) if callable(getattr(self, attr)) and not attr.startswith("__")]
            print("Wrapped functions:",self.meth)

            pass

    def matmul(self,u,v):
        return tf.matmul(u,v)
    
    def transpose(self,v):
        return tf.transpose(v)
    
    def transpose_batch(self,v):
        return tf.transpose(v,[0,2,1])
    
    def sqrt(self,x):
  
        return tf.math.sqrt(tf.cast(x, x.dtype))  
    
    def softmax(self,v):
        return tf.nn.softmax(v,axis=-1)
    
    def where(self,v):
        return tf.where(v)
    
    def matrix_exp(self,v):
        return tf.linalg.expm(v)
    
    def shape_last_dim(self,x):
        return tf.cast(tf.shape(x)[-1], x.dtype)
    
    def causal_mask(self,x):
        input_shape = tf.shape(x)
        batch_size, sequence_length = input_shape[0], input_shape[1]

        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)

        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))

        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
            )

        return tf.tile(mask, mult)
    
