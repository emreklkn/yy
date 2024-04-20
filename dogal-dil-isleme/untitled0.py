import tensorflow as tf

hello = tf.constant('hello')
print(hello.numpy())
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

hello = tf.constant('hello')
sess = tf.Session()
print(sess.run(hello))

