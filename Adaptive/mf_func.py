'''
@Multi-fidelity DNN for function approximation
@Author: Xuhui Meng
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import numpy as np
import matplotlib.pyplot as plt
from net import DNN 

np.random.seed(1234)
tf.set_random_seed(1234)
 
#exact low-fidelity function
def fun_lf(x):
    y = 0.5*(6*x - 2)**2*np.sin(12*x - 4) + 10*(x - 0.5) - 5
    return y

#exact high-fidelity function
def fun_hf(x):
    y = (6*x - 2)**2*np.sin(12*x - 4)
    return y

def callback(loss_):
    global step
    step += 1
    if step%100 == 0:
        print('Loss: %.3e'%(loss_))

def main():
    D = 1
    #NNs
    #low-fidelity NN
    layers_lf = [D] + 2*[20] + [1]
    #nonlinear correlation
    layers_hf_nl = [D+1] + 2*[10] + [1]
    #linear correlation
    layers_hf_l = [D+1] + [1]

    #low-fidelity training data
    x_lf = np.linspace(0, 1, 21).reshape((-1, 1))
    y_lf = fun_lf(x_lf)
    #high-fidelity training data
    x_hf = np.array([0., 0.4, 0.6, 1.0]).reshape((-1, 1))
    #low-fidelity training data at x_H
    y_lf_hf = fun_lf(x_hf)
    y_hf = fun_hf(x_hf)
    X_hf = np.hstack((x_hf, y_lf_hf))

    Xmin = x_lf.min(0)
    Xmax = x_lf.max(0)
    Ymin = y_lf.min(0)
    Ymax = y_lf.max(0)

    Xhmin = np.hstack((Xmin, Ymin))
    Xhmax = np.hstack((Xmax, Ymax))

    x_train_lf = tf.placeholder(shape=[None, D], dtype=tf.float32)
    y_train_lf = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    x_train_hf = tf.placeholder(shape=[None, D+1], dtype=tf.float32)
    y_train_hf = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    model = DNN()
    W_lf, b_lf = model.hyper_initial(layers_lf)
    W_hf_nl, b_hf_nl = model.hyper_initial(layers_hf_nl)
    W_hf_l, b_hf_l = model.hyper_initial(layers_hf_l)

    y_pred_lf = model.fnn(W_lf, b_lf, x_train_lf, Xmin, Xmax)
    y_pred_hf_nl = model.fnn(W_hf_nl, b_hf_nl, x_train_hf, Xhmin, Xhmax)
    y_pred_hf_l = model.fnn(W_hf_l, b_hf_l, x_train_hf, Xhmin, Xhmax)
    y_pred_hf = y_pred_hf_l + y_pred_hf_nl

    loss_l2 = 0.01*tf.add_n([tf.nn.l2_loss(w_) for w_ in W_hf_nl])
    loss_lf = tf.reduce_mean(tf.square(y_pred_lf - y_train_lf))
    loss_hf =  tf.reduce_mean(tf.square(y_pred_hf - y_train_hf))
    loss_hf_l =  tf.reduce_mean(tf.square(y_pred_hf_l - y_train_hf))
    loss = loss_lf + loss_hf + loss_hf_l
    train_adam = tf.train.AdamOptimizer().minimize(loss)
    train_lbfgs = ScipyOptimizerInterface(loss,
                                          method = 'L-BFGS-B', 
                                          options={'maxiter': 50000,
                                                   'ftol': 1.0*np.finfo(float).eps
                                                  })

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nmax = 5000
    loss_c = 1.0e-3
    loss_ = 1.0
    n = 0
    train_dict = {x_train_lf: x_lf, y_train_lf: y_lf, x_train_hf: X_hf, y_train_hf: y_hf}
    #optimization:
    #1. Train all NNs using Adam for nmax steps or until the loss is less than 10^-3
    while n < nmax and loss_ > loss_c:
        n += 1
        loss_, _, loss_lf_, loss_hf_ = sess.run([loss, train_adam, loss_lf, loss_hf], feed_dict=train_dict)
        if n%1000 == 0:
            print('n: %d, loss: %.3e, loss_lf: %.3e, loss_hf: %.3e'%(n, loss_, loss_lf_, loss_hf_))
    #Switch to LBFGS until convergence
    train_lbfgs.minimize(sess, feed_dict=train_dict, fetches=[loss], loss_callback=callback)

    #prediction
    x_test = np.linspace(0, 1, 1000).reshape((-1, 1))
    y_lf_ref = fun_lf(x_test)
    y_hf_ref = fun_hf(x_test)
    y_lf_test = sess.run(y_pred_lf, feed_dict={x_train_lf: x_test})
    X_test = np.hstack((x_test, y_lf_test))
    y_hf_test = sess.run(y_pred_hf, feed_dict={x_train_hf: X_test})

    plt.figure()
    plt.plot(x_lf, y_lf, 'go')
    plt.plot(x_test, y_lf_ref, 'k-')
    plt.plot(x_test, y_lf_test, 'r--')

    plt.plot(x_hf, y_hf, 'bo')
    plt.plot(x_test, y_hf_ref, 'k-')
    plt.plot(x_test, y_hf_test, 'c--')
#    plt.savefig('linear_func.png')
    plt.show()

if __name__ == '__main__':
    step = 0
    main()
