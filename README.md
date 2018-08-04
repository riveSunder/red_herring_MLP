# red_herring_MLP
Multilayer perceptrons built with <code>tf.layers</code> or <code>tf.matmul</code> and trained with <code>train_op.run<code> or <code>tf.contrib.learn.Estimator.fit</code>
  
I was recently assigned a programming assignment as part of the application process for a machine learning position, my first experience with this sort of homework that I understand is a common part of applying to software positions. While I’ll respect the confidentiality of the assignment itself (it was weird), I can talk about the study tips from the homework invitation email, as these essentially had nothing to do with the actual assignment.

Applicants were encouraged to bone up on multi-layer dense neural networks, aka multi-layer perceptrons, using TensorFlow and TensorBoard. To get ready for the assignment, I built two six-layer MLPs at different levels of abstraction: a lower-level MLP using explicit matrix multiplication and activation, and a higher-level MLP using <code>tf.layers</code> and <code>tf.contrib.learn</code>. I used the iris, wine, and digits datasets from scikit-learn as these are small enough to iterate over a lot of variations without taking too much time. Although the exercise didn’t end up being specifically useful to the coding assignment, I did get more familiar with using TensorBoard and <a href="https://www.tensorflow.org/api_docs/python/tf/summary"><code>tf.summary</code></a> commands. 

Although my intention was to design identical models using different tools, and despite using the same Adam optimizer for training, the higher-level abstracted model performed much better (often achieving 100% accuracy on the validation datasets) than the model built around <code>tf.matmul</code> operations. Being a curious sort I set out to find out what was leading to the performance difference and built two more models mixing <code>tf.layers</code>, <code>tf.contrib.learn</code>, and <code>tf.matmul</code>. 

In genetics research it’s common practice to determine relationships between genes and traits by breaking things until the trait disappears, than trying to restore the trait by externally adding specific genes back to compensate for the broken one. This would go fall under the terms "knockout" and "rescue," respectively, and I took a similar approach here. My main findings were:

<ul>
    <li>Replacing <code>tf.matmul</code>	operations with <code>tf.layers</code> didn’t have much effect. Changing dropout and other hyperparameters did not seem to effect the low-level and high-level models differently.</li>
    <li>"Knocking out" the use of <code>learn.Estimator.fit</code> from <code>tf.contrib.learn</code>and running the training optimizer directly led to significantly degraded performance of the <code>tf.layers</code> model.</li>
   <li>The model built around <code>tf.matmul</code> could be "rescued" by training with <code>learn.Estimator.fit</code>instead of <code>train_op.run</code>. </li>
    <li>The higher-level model using layers did generally perform a little better than the lower-level model, especially on the digits dataset.</li>
</ul>
