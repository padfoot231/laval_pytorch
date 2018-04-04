# laval_pytorch
mnist_train.py: It is pytorch code for training the MNIST data, the program saves the pickel files with only paramete with the use of "state_dict"( method 1 ) through this we save only the parameters instead the whole model.

checkpoint_mnist_train.py : It is training the mnist data but saving the entire model it allows model to be used by someone else without the access to your code.

checkpoint_mnist_test.py : It is pytorch code of loading bianry files with parameters ( method 2 )

checkpoint_mnist_test.py : It is pytorch code of loading bianry files with entire model ( method 1 )

hook.py : shows the use of register_hook() to print certain variables gradients

Gradietn_train.py : pytorch code to train mnist model and to print the gradient just before 1st FC layer

Data: is enclosed in zipped file "train&test data.tar.gz"
