import tensorflow as tf
# from tensorflow import keras  # 错误导入方式，错误原因不明，错误结果：无keras类的提示
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, optimizers, metrics

# 加载数据
(xs, ys), _ = datasets.mnist.load_data()
# 输出数据形状大小
print("datasets:", xs.shape, ys.shape)

batch_size = 32

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255
db = tf.data.Dataset.from_tensor_slices((xs, ys))
db = db.batch(batch_size).repeat(30)

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])
model.build(input_shape=(4, 28 * 28))
model.summary()

optimizer = optimizers.SGD(learning_rate=0.001)
acc_meter = metrics.Accuracy()

for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        # 打平操作，[b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28 * 28))
        # Step1. 得到模型输出output [b, 784] => [b, 10]
        out = model(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # 计算差的平方和，[b, 10]
        loss = tf.square(out - y_onehot)
        # 计算每个样本的平均误差，[b]
        loss = tf.reduce_sum(loss) / x.shape[0]

        acc_meter.update_state(tf.argmax(out, axis=1), y)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 200 == 0:
            print(
                step,
                'loss:',
                float(loss),
                'acc:',
                acc_meter.result().numpy())
            acc_meter.reset_states()
