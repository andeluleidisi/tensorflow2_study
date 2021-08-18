import tensorflow as tf
import matplotlib.pyplot as plt
import math

def fun(x):
    """

    :param x: [b,2]
    :return:
    """
    z = tf.math.sin(x[...,0]) + tf.math.sin(x[...,1])
    return z

x = tf.linspace(0.,2*math.pi,500)
y = tf.linspace(0.,2*math.pi,500)
# [50, 50]
point_x,point_y = tf.meshgrid(x,y)
# [50, 50, 2]
point = tf.stack(values=[point_x,point_y],axis=2)  # stack只能用于tensor合并
z = fun(point)
print(z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(x, y, z)
plt.colorbar()
plt.show()
