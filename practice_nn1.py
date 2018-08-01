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

# user-definable model attributes
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate',1e-3,"""Learning rate for classification""")

tf.app.flags.DEFINE_float('dropout_rate',0.0,"""dropout rate""")
tf.app.flags.DEFINE_integer('layer_size',64,"""starting hidden layer size""")
tf.app.flags.DEFINE_integer('unit_divisor',1,"""fraction of prior layer's units in each subsequent layer. E.g. a factor of 2 halves the units in each layer""")
tf.app.flags.DEFINE_integer('max_epochs',150,"""Number of epochs to train""")
tf.app.flags.DEFINE_integer('batch_size',50,"""batch size""")
tf.app.flags.DEFINE_string('graph_dir',"./graph/run1","""Directory for storing tensorboard summaries""")
tf.app.flags.DEFINE_string('dataset',"iris","""Datset for training""")

learning_rate = FLAGS.learning_rate
unit_divisor = FLAGS.unit_divisor
layer_size = FLAGS.layer_size
max_epochs = FLAGS.max_epochs
batch_size = FLAGS.batch_size
dropout_rate = FLAGS.dropout_rate
graph_dir = FLAGS.graph_dir
dataset = FLAGS.dataset


if (layer_size <=16):
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

X = X.reshape([X.shape[0],1,number_features])   

# set up variablse
with tf.variable_scope("lucky_MLP"):
    w0 = tf.Variable(tf.random_normal([1,number_features,layer0_size], stddev=0.125),name="w0")
    w1 = tf.get_variable("w1",initializer=tf.random_normal([1,layer0_size,layer1_size], stddev=0.125))
    w2 = tf.get_variable("w2",initializer=tf.random_normal([1,layer1_size,layer2_size], stddev=0.125))
    w3 = tf.get_variable("w3",initializer=tf.random_normal([1,layer3_size,layer4_size], stddev=0.125))
    w4 = tf.get_variable("w4",initializer=tf.random_normal([1,layer4_size,layer5_size], stddev=0.125))
    w5 = tf.get_variable("w5",initializer=tf.random_normal([1,layer5_size,layer6_size], stddev=0.125))
    w6 = tf.get_variable("w6",initializer=tf.random_normal([1,layer5_size,number_classes], stddev=0.125))

    starting_bias = 1e-3
    b0 = tf.get_variable("b0",initializer=starting_bias)
    b1 = tf.get_variable("b1",initializer=starting_bias)
    b2 = tf.get_variable("b2",initializer=starting_bias)
    b3 = tf.get_variable("b3",initializer=starting_bias)
    b4 = tf.get_variable("b4",initializer=starting_bias)
    b5 = tf.get_variable("b5",initializer=starting_bias)
    b6 = tf.get_variable("b6",initializer=starting_bias)

    # 
inputs = tf.placeholder("float",[None,1,number_features],name="inputs")
targets = tf.placeholder("float",[None,number_classes],name="targets")
training_mode = tf.placeholder("bool",name="training_mode")


def nn_MLP(inputs,targets,training_mode):

    inputs =  tf.reshape(inputs,[-1,1,number_features])
                             
    layer_0 = (tf.nn.relu(tf.nn.conv1d(inputs,w0,1,"VALID")+b0))
    dropout_0 =  tf.layers.dropout(inputs=layer_0,
                               rate=dropout_rate,
                               training = training_mode)

    layer_1 = (tf.nn.relu(tf.nn.conv1d(dropout_0,w1,1,"VALID")+b1))	
    dropout_1 =  tf.layers.dropout(inputs=layer_1,
                               rate=dropout_rate,
                               training = training_mode)

    layer_2 = (tf.nn.relu(tf.nn.conv1d(dropout_1,w2,1,"VALID")+b2))
    dropout_2 =  tf.layers.dropout(inputs=layer_2,
                               rate=dropout_rate,
                               training =  training_mode)

    layer_3 = (tf.nn.relu(tf.nn.conv1d(dropout_2,w3,1,"VALID")+b3))
    dropout_3 =  tf.layers.dropout(inputs=layer_3,
                               rate=dropout_rate,
                               training = training_mode)

    layer_4 = (tf.nn.relu(tf.nn.conv1d(dropout_3,w4,1,"VALID")+b4))
    dropout_4 =  tf.layers.dropout(inputs=layer_4,
                               rate=dropout_rate,
                               training = training_mode)
                               
    layer_5 = (tf.nn.relu(tf.nn.conv1d(dropout_4,w5,1,"VALID")+b5))
    logits = tf.nn.conv1d(layer_5,w6,1,"VALID")+b6

    print(logits.shape)

    logits = tf.reshape(logits,[-1,number_classes])

    predictions = tf.nn.softmax(logits)

    print(layer_0.shape)
    print(layer_1.shape)
    print(layer_2.shape)
    print(layer_3.shape)
    print(layer_4.shape)
    print(layer_5.shape)
    print(logits.shape)
        
    tf.summary.histogram("weights_2",w2)
    tf.summary.histogram("weights_3",w3)
    tf.summary.histogram("weights_4",w4)
    tf.summary.histogram("weights_5",w5)
    tf.summary.histogram("weights_6",w6)
    tf.summary.image("layer_0_activations",tf.reshape(layer_0,[-1, dimX, dimY,1]))
    tf.summary.image("layer_1_activations",tf.reshape(layer_1,[-1, dimX, dimY,1]))
    tf.summary.image("layer_2_activations",tf.reshape(layer_2,[-1, dimX, dimY,1]))
    tf.summary.image("layer_3_activations",tf.reshape(layer_3,[-1, dimX, dimY,1]))
    tf.summary.image("layer_5_activations",tf.reshape(layer_4,[-1, dimX, dimY,1]))
    tf.summary.image("layer_6_activations",tf.reshape(layer_5,[-1, dimX, dimY,1]))

    return predictions


#return predictions

predictions = nn_MLP(inputs,targets,training_mode)

classification_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels = targets, logits = predictions))


classification_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                 beta1=0.9,
                                 beta2 = 0.999,
                                 epsilon=1e-08,
                                 use_locking=False,
                                 name='adam').minimize(classification_loss)

correct_predictions = tf.equal(tf.argmax(predictions,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
tf.summary.scalar("accuracy",accuracy)
tf.summary.scalar("loss",classification_loss)

tf.summary.histogram("predictions",tf.argmax(predictions,axis=1))

tf.summary.histogram("weights_0_grad",tf.gradients(classification_loss,[w0])[0])
tf.summary.histogram("weights_1_grad",tf.gradients(classification_loss,[w1])[0])
tf.summary.histogram("weights_2_grad",tf.gradients(classification_loss,[w2])[0])
tf.summary.histogram("weights_3_grad",tf.gradients(classification_loss,[w3])[0])
tf.summary.histogram("weights_4_grad",tf.gradients(classification_loss,[w4])[0])
tf.summary.histogram("weights_5_grad",tf.gradients(classification_loss,[w5])[0])
tf.summary.histogram("weights_6_grad",tf.gradients(classification_loss,[w6])[0])


merge = tf.summary.merge_all()

def main():
	
	
	t0 = time.time()
	#losses = []
	# set up one hot labels for targets
	number_entries = Y.shape[0]
	onehot_Y = np.zeros((number_entries,number_classes))
	for counter in range(number_entries):
		onehot_Y[counter,Y[counter]] = 1
    
	
	# divvy up training and data into training, validation, and test
	np.random.seed(random_seed)
	np.random.shuffle(X)
	np.random.seed(random_seed)
	np.random.shuffle(onehot_Y)
	validation_size = int(0.1*number_entries)

	validation_X = X[0:validation_size,...]
	validation_Y = onehot_Y[0:validation_size,:]

	test_X = X[validation_size:validation_size+validation_size,...]
	test_Y = onehot_Y[validation_size:validation_size+validation_size,:]
	
	train_X = X[validation_size+validation_size:number_entries,...]
	train_Y = onehot_Y[validation_size+validation_size:number_entries,:]

    
	with tf.Session() as sess:
		#tf.global_variables_initializer()
		tf.initialize_all_variables().run()

		train_writer = tf.summary.FileWriter(graph_dir+"layersize"+str(layer_size)+"dropout"+str(int(dropout_rate*100))+"div"+str(unit_divisor)+dataset+"/", sess.graph)
		print("./"+dataset+"/"+graph_dir)

		print("start training")
		for epoch_counter in range(max_epochs):
			for batch_counter in range(0,batch_size,len(train_X)):

				# iterate through training data
				inputs_ = train_X[batch_counter:batch_counter+batch_size]
				targets_ = train_Y[batch_counter:batch_counter+batch_size]

				classification_train_op.run(feed_dict = {inputs: inputs_, targets: targets_, training_mode: True})
			# display current loss
			if (0):
				# subsample training set if too large
				inputs_ = train_X[0:batch_size]
				targets_ = train_Y[0:batch_size]
			else:
				inputs_ = train_X
				targets_ = train_Y

			if(epoch_counter % 50 == 0):
				
				

				# summarize accuracy and loss for reporting
				train_accuracy = accuracy.eval(feed_dict={inputs: inputs_, targets: targets_,training_mode: False})

				val_accuracy = accuracy.eval(feed_dict={inputs: validation_X, targets: validation_Y,training_mode: False})
				train_loss = sess.run(classification_loss,feed_dict={inputs: inputs_, targets: targets_,training_mode: False})
				val_loss = sess.run(classification_loss,feed_dict={inputs: validation_X, targets: validation_Y,training_mode: False})
				print("validation shape",validation_X.shape)
				summary = sess.run(merge,feed_dict={inputs: validation_X, targets: validation_Y, training_mode: False})
				train_writer.add_summary(summary,epoch_counter)#,step=tf.train.get_global_step())

				elapsed = time.time() - t0

				print("training / validation loss epoch %i : %.3e / %.3e "%(epoch_counter,train_loss,val_loss))
				print("training / validation accuracy: %.3e / %.3e"%(train_accuracy,val_accuracy))
				print("time elapse: %.2f"%elapsed)


				if(0):
					# Hints say to "be able to get variables from  a graph" and here are a few ways to do that: 
					with tf.variable_scope("lucky_MLP",reuse=True):
						# note that variables have to be initialized with get_variable and reuse=True has to be set in the same scope as they were initialized in
						test = tf.get_variable("b0")
						test2 = tf.get_variable("w0")
						print("bias 0 and mean weights_0 = ",test.eval(sess),np.mean(test2.eval(sess)))
					# get values for layers or weights with sess.run
					print("layer 3 mean values",np.mean(sess.run(layer_3,feed_dict = {inputs: inputs_, targets: targets_, training_mode: False})))
					print("weights 3 mean values",np.mean(sess.run(w3,feed_dict = {inputs: inputs_, targets: targets_, training_mode: False})))
 				
	
main()
#if __name__ == "__main__":
#    tf.app.run()
	


