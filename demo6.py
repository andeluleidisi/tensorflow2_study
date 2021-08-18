import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 数据预处理函数
def preprocess(x,y):
    '''

    :param x:
    :param y:
    :return:
    '''
    # 转换x的格式为float32,并归一化
    x = tf.cast(x,dtype=tf.float32) / 255.
    # 转换x的格式
    x = tf.reshape(x,shape=[-1,28*28])
    # 转换y的格式为int32
    y = tf.cast(y,dtype=tf.int32)
    # 转换为one hot码
    y = tf.one_hot(y,depth=10)

    return x,y

# 数据加载函数
def load_data():
    '''

    :return:
    '''
    (x,y),(x_test,y_test) = datasets.mnist.load_data()
    print(x.shape,y.shape,x_test.shape,y_test.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x,y))
    test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))

    train_db = train_db.shuffle(1000).batch(128).map(map_func=preprocess).repeat(10)
    test_db = test_db.shuffle(1000).batch(128).map(map_func=preprocess)

    return train_db,test_db

def main():
    # 定义学习率
    lr = 1e-3

    w1 , b1 = tf.Variable(tf.random.truncated_normal(shape=[784,512],stddev=0.1)),tf.Variable(tf.zeros([512]))
    w2 , b2 = tf.Variable(tf.random.truncated_normal(shape=[512, 256], stddev=0.1)),tf.Variable(tf.zeros([256]))
    w3 , b3 = tf.Variable(tf.random.truncated_normal(shape=[256, 10], stddev=0.1)),tf.Variable(tf.zeros([10]))

    train_db, test_db = load_data()
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            out = h2 @ w3 + b3

            # [b, 10] - [b, 10] => [b, 10] => [b]
            loss = tf.reduce_mean(tf.square(y - out),axis=1)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(target=loss,sources=[w1,b1,w2,b2,w3,b3])

        # 原地更新参数
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr*g)

        # print
        if step % 100 == 0:
            print(step, 'loss:', float(loss))

        # evaluate
        if step % 500 == 0:
            total, total_correct = 0., 0

            for step, (x, y) in enumerate(test_db):
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct / total)


main()