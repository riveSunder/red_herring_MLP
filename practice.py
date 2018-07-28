import numpy as np
import matplotlib.pyplot as plt

import time 


"""A low level MLP classifier implemented in Tensorflow"""
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
tf.app.flags.DEFINE_string('graph_dir',"./graph","""Directory for storing tensorboard summaries""")

learning_rate = FLAGS.learning_rate
unit_divisor = FLAGS.unit_divisor
layer_size = FLAGS.layer_size
max_epochs = FLAGS.max_epochs
batch_size = FLAGS.batch_size
dropout_rate = FLAGS.dropout_rate
graph_dir = FLAGS.graph_dir

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
[iris_data,iris_targets] = datasets.load_iris(return_X_y=True)

X = iris_data
Y = iris_targets

# dim 0 is data points, dim 1 is features
number_features = X.shape[1]
number_classes = 3 # 3 classes for iris dataset

# weights
w0 = tf.Variable(tf.truncated_normal([number_features,layer0_size],stddev=0.05),name='w0')
w1 = tf.Variable(tf.truncated_normal([layer1_size,layer2_size],stddev=0.05),name='w1')
w2 = tf.Variable(tf.truncated_normal([layer2_size,layer3_size],stddev=0.05),name='w2')
w3 = tf.Variable(tf.truncated_normal([layer3_size,layer4_size],stddev=0.05),name='w3')
w4 = tf.Variable(tf.truncated_normal([layer4_size,layer5_size],stddev=0.05),name='w4')
w5 = tf.Variable(tf.truncated_normal([layer5_size,layer6_size],stddev=0.05),name='w5')
w6 = tf.Variable(tf.truncated_normal([layer6_size,number_classes],stddev=0.05),name='w6')

# biases
starting_bias = 1e-3
b0 = starting_bias 
b1 = starting_bias 
b2 = starting_bias 
b3 = starting_bias 
b4 = starting_bias 
b5 = starting_bias 
b6 = starting_bias 

# 
inputs = tf.placeholder("float",[None,number_features],name="inputs")
targets = tf.placeholder("float",[None,number_classes],name="targets")
training_mode = tf.placeholder("bool",name="training_mode")


def evaluate_accuracy(y,predictions):
    """Function for determining accuracy of logits vs targets"""
    # y and pred must be the same size
    if( not (np.shape(y) == np.shape(predictions))):
        print("Error (shapes don't match)")
    m = np.shape(y)[0]
    accuracy = np.mean(1.*np.argmax(y,axis=1) == np.argmax(predictions,axis=1))
    return accuracy

def lucky_MLP(inputs,targets,training_mode):
	# reshape of inputs (not required, but may add noise here later)
	inputs =  tf.reshape(inputs,[-1,number_features])
	
	layer_0 = tf.nn.dropout(tf.nn.relu(tf.matmul(inputs,w0)+b0),(1-dropout_rate),name="layer_0")
	layer_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_0,w1)+b1),(1-dropout_rate),name="layer_1")
	layer_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_1,w2)+b2),(1-dropout_rate),name="layer_2")
	layer_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_2,w3)+b3),(1-dropout_rate),name="layer_3")
	layer_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_3,w4)+b4),(1-dropout_rate),name="layer_4")
	layer_5 = tf.nn.dropout((tf.matmul(layer_4,w5)+b5),(1-dropout_rate),name="layer_5")

	predictions = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_5,w6)+b6),(1),name="predictions")

	tf.summary.histogram("layer_0",layer_0)
	tf.summary.histogram("layer_1",layer_0)
	tf.summary.histogram("layer_2",layer_0)
	tf.summary.histogram("layer_3",layer_0)
	tf.summary.histogram("layer_4",layer_0)
	tf.summary.histogram("layer_5",layer_0)

	tf.summary.histogram("predictions",predictions)
	
	return predictions

predictions = lucky_MLP(inputs,targets,training_mode)

classification_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels = targets, logits = predictions))


classification_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                 beta1=0.9,
                                 beta2 = 0.999,
                                 epsilon=1e-08,
                                 use_locking=False,
                                 name='adam').minimize(classification_loss)


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

	validation_X = X[0:validation_size,:]
	validation_Y = onehot_Y[0:validation_size,:]
	test_X = X[validation_size:validation_size+validation_size,:]
	test_Y = onehot_Y[validation_size:validation_size+validation_size,:]
	
	train_X = X[validation_size+validation_size:number_entries,:]
	train_Y = onehot_Y[validation_size+validation_size:number_entries,:]

	with tf.Session() as sess:
		#tf.global_variables_initializer()
		tf.initialize_all_variables().run()
		
		train_writer = tf.summary.FileWriter(graph_dir, sess.graph)

		print("start training")
		for epoch_counter in range(max_epochs):
			for batch_counter in range(0,batch_size,len(train_X)):

				# iterate through training data
				input_ = train_X[batch_counter:batch_counter+batch_size]
				targets_ = train_Y[batch_counter:batch_counter+batch_size]

				sess.run(classification_train_op, feed_dict = {inputs: input_, targets: targets_, training_mode: True})
			# display current loss
			if (0):
				# subsample training set if too large
				input_ = train_X[0:batch_size]
				targets_ = train_Y[0:batch_size]
			else:
				input_ = train_X
				targets_ = train_Y

			[train_loss,train_predictions] = sess.run([classification_loss,predictions], feed_dict={inputs: input_, targets: targets_, training_mode: False})
			[val_loss,val_predictions] = sess.run([classification_loss,predictions], feed_dict={inputs: validation_X, targets: validation_Y, training_mode: False})			
			
			train_accuracy = evaluate_accuracy(targets_,train_predictions)
			val_accuracy = evaluate_accuracy(validation_Y,val_predictions)
			
			# summarize accuracy and loss for tensorboard
			
			tf.summary.scalar("train_accuracy",train_accuracy)
			tf.summary.scalar("val_accuracy",val_accuracy)
			tf.summary.scalar("train_loss",train_loss)
			tf.summary.scalar("val_loss",val_loss)

			merge = tf.summary.merge_all()
			summary = sess.run(merge,feed_dict={inputs: validation_X, targets: validation_Y, training_mode: False})
			train_writer.add_summary(summary,epoch_counter)

			elapsed = time.time() - t0
			print("training / validation loss epoch %i : %.3e / %.3e "%(epoch_counter,train_loss,val_loss))
			print("training / validation accuracy: %.3e / %.3e"%(train_accuracy,val_accuracy))
			print("time elapse: %.2f"%elapsed)
 				
	
main()
#if __name__ == "__main__":
#    tf.app.run()
	


