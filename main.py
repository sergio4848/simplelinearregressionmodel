import tensorflow as tf
import numpy as np

# Generate some data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Create a linear regression model
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))

# Define the loss function
def mse_loss():
  y = W * x_data + b
  loss = tf.reduce_mean(tf.square(y - y_data))
  return loss

# Optimize the model
optimizer = tf.keras.optimizers.Adam()
for step in range(5000):
  optimizer.minimize(mse_loss, var_list=[W, b])
  if step % 500 == 0:
    print(step, W.numpy(), b.numpy())

# Print the final results
print("W:", W.numpy())
print("b:", b.numpy())
