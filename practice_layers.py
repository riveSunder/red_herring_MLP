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


def layers_MLP(inputs,targets,mode):
	# define the graph using tf.layers 
	inputs =  tf.reshape(inputs,[-1,number_features])

	layer_0 = tf.layers.dense(inputs=inputs,
                             units=layer_size,
                             activation=tf.nn.relu)
	dropout_0 =  tf.layers.dropout(inputs=layer_0,
                               rate=dropout_rate,
                               training = mode == learn.ModeKeys.TRAIN)

	layer_1 = tf.layers.dense(inputs=dropout_0,
                             units=layer_size,
                             activation=tf.nn.relu)	
	dropout_1 =  tf.layers.dropout(inputs=layer_1,
                               rate=dropout_rate,
                               training = mode == learn.ModeKeys.TRAIN)

	layer_2 = tf.layers.dense(inputs=dropout_1,
                             units=layer_size,
                             activation=tf.nn.relu)
	dropout_2 =  tf.layers.dropout(inputs=layer_2,
                               rate=dropout_rate,
                               training =  mode == learn.ModeKeys.TRAIN)

	layer_3 = tf.layers.dense(inputs=dropout_2,
                             units=layer_size,
                             activation=tf.nn.relu)
	dropout_3 =  tf.layers.dropout(inputs=layer_3,
                               rate=dropout_rate,
                               training = mode == learn.ModeKeys.TRAIN)

	layer_4 = tf.layers.dense(inputs=dropout_3,
                             units=layer_size,
                             activation=tf.nn.relu)
	dropout_4 =  tf.layers.dropout(inputs=layer_4,
                               rate=dropout_rate,
                               training = mode == learn.ModeKeys.TRAIN)

	layer_5 = tf.layers.dense(inputs=dropout_4,
                             units=layer_size,
                             activation=tf.nn.relu)

	logits = tf.layers.dense(inputs=layer_5,
                             units=number_classes,
                             activation=None)
	
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
	MLP_classifier = learn.Estimator(model_fn = layers_MLP,\
		model_dir = graph_dir,\
		config=tf.contrib.learn.RunConfig(save_checkpoints_secs=3))

	#Assign metrics
	metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,prediction_key="classes")}

	
	tensors_to_log = {"probabilities": "softmaxTensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,every_n_iter = 1)
	
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
		early_stopping_rounds=6400)

	
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
	


