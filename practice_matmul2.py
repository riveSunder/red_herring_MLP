import numpy as np
import matplotlib.pyplot as plt

import time 


"""A low level MLP classifier implemented in Tensorflow. 'Low level' as in it uses explicit activation and matmul operations instead of tf.layers.dense. The optimizer is still abstracted by TF with very little effort from yours truly. """
# Have tensorflow 1.4 and python 3.x installed
# know how to make a multi-level perceptron classifier
# know how to get variable from a graph
# Review tensorboard 
# I'm in the process of adopting PEP guidelines, so I'll try to use_underscores_for_variables unless I accidentally and useCamelCase

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)

# user-definable model attributes
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate',1e-3,"""Learning rate for classification""")

tf.app.flags.DEFINE_float('dropout_rate',0.0,"""dropout rate""")
tf.app.flags.DEFINE_integer('layer_size',64,"""starting hidden layer size""")
tf.app.flags.DEFINE_integer('unit_divisor',1,"""fraction of prior layer's units in each subsequent layer. E.g. a factor of 2 halves the units in each layer""")
tf.app.flags.DEFINE_integer('max_steps',150,"""Number of epochs to train""")
tf.app.flags.DEFINE_integer('batch_size',64,"""batch size""")
tf.app.flags.DEFINE_string('graph_dir',"./layersmodel/run1","""Directory for storing tensorboard summaries""")
tf.app.flags.DEFINE_string('dataset',"iris","""Datset for training""")

learning_rate = FLAGS.learning_rate
unit_divisor = FLAGS.unit_divisor
layer_size = FLAGS.layer_size
max_steps = FLAGS.max_steps
batch_size = FLAGS.batch_size
dropout_rate = FLAGS.dropout_rate
graph_dir = FLAGS.graph_dir
dataset = FLAGS.dataset

if (layer_size ==1):
	layer_size = 1
	dimX = 1
	dimY = 1
elif (layer_size <=2):
	layer_size = 2
	dimX = 4
	dimY = 2
elif (layer_size <=4):
	layer_size = 4
	dimX = 2
	dimY = 2
elif (layer_size <=8):
	layer_size = 8
	dimX = 4
	dimY = 2
elif (layer_size <=16):
	layer_size = 16
	dimX = 4
	dimY = 4
elif (layer_size <= 32):
	layer_size = 32
	dimX = 8
	dimY = 4
elif (layer_size <= 64):
	layer_size = 64
	dimX = 8
	dimY = 8
elif (layer_size <= 128):
	layer_size=128
	dimX = 16
	dimY = 8
elif (layer_size <= 256):
	layer_size = 256
	dimX = 16
	dimY = 16
elif (layer_size <= 512):
	layer_size = 512
	dimX = 32
	dimY = 16
elif (layer_size <= 1024):
	layer_size = 1024
	dimX = 32
	dimY = 32
else:
	layer_size=2048
	dimX = 64
	dimY = 32

layer0_size = layer_size #64
layer1_size = int(layer0_size/unit_divisor) #32
layer2_size = int(layer1_size/unit_divisor) #16
layer3_size = int(layer2_size/unit_divisor) #8
layer4_size = int(layer3_size*unit_divisor) #16
layer5_size = int(layer4_size*unit_divisor) #32
layer6_size = int(layer5_size*unit_divisor) #64 

random_seed = 42 # I wonder how much the ubiquity of using this seed influences ML research? 

# load the iris dataset from sklearn datasets
import sklearn.datasets as datasets
if(dataset=="iris"):
	# load iris dataset 4 features 3 classes 150 samples
	print("loading iris dataset")
	[iris_data,iris_targets] = datasets.load_iris(return_X_y=True)
	X = iris_data
	Y = iris_targets
elif(dataset=="wine"):
	# load wine dataset 13 features 3 classes 178 samples 
	print("loading wine dataset")
	[wine_data,wine_targets] = datasets.load_wine(return_X_y=True)
	X = wine_data
	Y = wine_targets
elif(dataset=="digits"):
	# load digits dataset 64 features 10 classes 1797 samples
	print("loading digits dataset") 
	[digits_data,digits_targets] = datasets.load_digits(return_X_y=True)
	X = digits_data
	Y = digits_targets
	print(X.shape)
# dim 0 is data points, dim 1 is features
number_features = X.shape[1]
number_classes = int(np.max(Y))+1# 3 classes for iris dataset
print(number_classes)

# 
inputs = tf.placeholder("float",[None,number_features],name="inputs")
targets = tf.placeholder("float",[None,number_classes],name="targets")
mode = tf.placeholder("bool",name="mode")


#with tf.variable_scope("lucky_MLP"):
# weights
#g = tf.graph


def matmul_MLP(inputs,targets,mode):
	
    w0 = tf.get_variable("w0",initializer=tf.truncated_normal([number_features,layer0_size],stddev=0.05))
    w1 = tf.get_variable("w1",initializer=tf.truncated_normal([layer1_size,layer2_size],stddev=0.05))
    w2 = tf.get_variable("w2",initializer=tf.truncated_normal([layer2_size,layer3_size],stddev=0.05))
    w3 = tf.get_variable("w3",initializer=tf.truncated_normal([layer3_size,layer4_size],stddev=0.05))
    w4 = tf.get_variable("w4",initializer=tf.truncated_normal([layer4_size,layer5_size],stddev=0.05))
    w5 = tf.get_variable("w5",initializer=tf.truncated_normal([layer5_size,layer6_size],stddev=0.05))
    w6 = tf.get_variable("w6",initializer=tf.truncated_normal([layer6_size,number_classes],stddev=0.05))

    # biases
    starting_bias = 1e-3
    b0 = tf.get_variable("b0",initializer=starting_bias)
    b1 = tf.get_variable("b1",initializer=starting_bias)
    b2 = tf.get_variable("b2",initializer=starting_bias)
    b3 = tf.get_variable("b3",initializer=starting_bias)
    b4 = tf.get_variable("b4",initializer=starting_bias)
    b5 = tf.get_variable("b5",initializer=starting_bias)
    b6 = tf.get_variable("b6",initializer=starting_bias)
    #def lucky_MLP(inputs,targets,training_mode):
    # reshape of inputs (not required, but may add noise here later)
    inputs =  tf.cast(tf.reshape(inputs,[-1,number_features]),tf.float32)

    layer_0 = tf.nn.dropout(tf.nn.relu(tf.matmul(inputs,w0)+b0),(1-dropout_rate),name="layer_0")
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_0,w1)+b1),(1-dropout_rate),name="layer_1")
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_1,w2)+b2),(1-dropout_rate),name="layer_2")
    layer_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_2,w3)+b3),(1-dropout_rate),name="layer_3")
    layer_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_3,w4)+b4),(1-dropout_rate),name="layer_4")
    layer_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_4,w5)+b5),(1-dropout_rate),name="layer_5")

    logits = tf.nn.dropout(tf.nn.softmax(tf.matmul(layer_5,w6)+b6),(1),name="predictions")


    if (1):# mode != learn.ModeKeys.INFER:
	    one_hot_labels = tf.one_hot(indices = tf.cast(targets,tf.int32),depth=number_classes) 
	    loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels,
			                          logits = logits)

	    tf.summary.scalar('cross_entropy', loss)

    if (1): #mode == learn.ModeKeys.TRAIN:
	    #trainOp = tf.train.MomentumOptimizer(lR,mom).minimize(loss,global_step = tf.contrib.framework.get_global_step())

	    train_opAdam = tf.train.AdamOptimizer(\
		    learning_rate=learning_rate,beta1=0.9,\
		    beta2 = 0.999,epsilon=1e-08,\
		    use_locking=False,name='Adam').minimize(\
			    loss,global_step = tf.contrib.framework.get_global_step())

	    train_opSGD = tf.contrib.layers.optimize_loss(\
		    loss = loss,\
		    global_step = tf.contrib.framework.get_global_step(),\
		    learning_rate = learning_rate,
		    optimizer = "SGD")
    if(dataset=="digits"):
	    tf.summary.image('input_image', tf.reshape(inputs,[-1,8,8,1]))
	    tf.summary.image('layer_0',tf.reshape(layer_0,[-1,dimX,dimY,1]))
	    tf.summary.image('layer_1',tf.reshape(layer_1,[-1,dimX,dimY,1]))
	    tf.summary.image('layer_2',tf.reshape(layer_2,[-1,dimX,dimY,1]))
	    tf.summary.image('layer_3',tf.reshape(layer_3,[-1,dimX,dimY,1]))
	    tf.summary.image('layer_4',tf.reshape(layer_4,[-1,dimX,dimY,1]))
	    tf.summary.image('layer_5',tf.reshape(layer_5,[-1,dimX,dimY,1]))
	
    predictions = {\
            "classes": tf.argmax(\
        	input=logits, axis=1),\
        	"probabilities":tf.nn.softmax(logits, name = "softmaxTensor")\
		    }


    # attach summaries for tensorboad https://www.tensorflow.org/get_started/summaries_and_tensorboard

    return model_fn_lib.ModelFnOps(\
	    mode=mode,\
	    predictions=predictions,\
	    loss=loss, train_op=train_opAdam)\

init = tf.global_variables_initializer() # deprecated ....

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.

#sess = tf.Session()
#sess.run(init)


def main(unused_argv):
	# Divvy up the data train/validation/test

	#losses = []
	# set up one hot labels for targets
	number_entries = Y.shape[0]
	if(0):	
		onehot_Y = np.zeros((number_entries,number_classes))
		for counter in range(number_entries):
			onehot_Y[counter,Y[counter]] = 1
	else:
		onehot_Y = Y

	# divvy up training and data into training, validation, and test
	np.random.seed(random_seed)
	np.random.shuffle(X)
	np.random.seed(random_seed)
	np.random.shuffle(onehot_Y)
	validation_size = int(0.1*number_entries)

	validation_X = X[0:validation_size,:]
	validation_Y = onehot_Y[0:validation_size]
	test_X = X[validation_size:validation_size+validation_size,:]
	test_Y = onehot_Y[validation_size:validation_size+validation_size]
	
	train_X = X[validation_size+validation_size:number_entries,:]
	train_Y = onehot_Y[validation_size+validation_size:number_entries]

	
	# Define the estimator
	MLP_classifier = learn.Estimator(model_fn = matmul_MLP,\
		model_dir = graph_dir,\
		config=tf.contrib.learn.RunConfig(save_checkpoints_secs=30))

	#Assign metrics
	metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,prediction_key="classes")}

	
	tensors_to_log = {"probabilities": "softmaxTensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,every_n_iter = 10)
	
	validation_metrics = {\
		"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,prediction_key="classes"),
		"precision":\
			tf.contrib.learn.MetricSpec(\
			metric_fn=tf.contrib.metrics.streaming_precision,\
			prediction_key=tf.contrib.learn.PredictionKey.CLASSES),\
		"recall":\
			tf.contrib.learn.MetricSpec(\
			metric_fn=tf.contrib.metrics.streaming_recall,\
			prediction_key=tf.contrib.learn.PredictionKey.CLASSES)\
		}

    #evalua during training and stop early if necessary
	
	validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
		validation_X,
		validation_Y,
		every_n_steps=1,
		metrics=validation_metrics,
		early_stopping_metric="accuracy",
		early_stopping_metric_minimize=False,
		early_stopping_rounds=16400)

	
	t0 = time.time()
	print("Begin training model with layer size %i"%layer_size)
	mode=True
	MLP_classifier.fit(x=train_X,
		    y=train_Y,
		    batch_size = batch_size,
		    steps = max_steps, # Steps DNE 
		    monitors = [validation_monitor])

    
	print("elapsed time: ",(time.time()-t0))
	# Evaluate model and display results
	eval_results = MLP_classifier.evaluate(x=train_X,
		                          y=train_Y,
		                          metrics=metrics)
	print("Validation results", eval_results)


if __name__ == "__main__":
    tf.app.run()
	


