### ***Assignment 4: Arnab K Ganguly, Batch 6 of EIP***2 

##### Submission Links:

| <u>Final Code with output</u>: https://drive.google.com/file/d/1FU6ru4Vo0BsAi5hDl_kOsiN8ENCBHCAk/view?usp=sharing |
| ------------------------------------------------------------ |
| <u>Latest checkpoint at 91% validation accuracy</u>: https://drive.google.com/file/d/1BIlGDS5LCVjjCstPZfUDddj3taUmO25M/view?usp=sharing |

| Final model salient features: 4 dense block network with compression and dropout, utilizing a l2 regularization scheme. Used nesterov momentum of 0.9 within a SGD optimizer. 940K parameters, Image augmentation using translation and horizontal flips |
| ------------------------------------------------------------ |
| ***Model training observations from experiments and lessons learnt in Section # 5 below*** |
| Experiments carried out: (1) Use of kernel max_norm constraint (2) Use of kernel regularizer                   (3) Use of data augmentation  (4) Use of different learning rates using a scheduler as well as reduce on plateau (5) Mixed training with the use of augmented as well as non-augmented images                          (6) Use of nesterov momentum within SGD optimizer (7) Use of different capacity models starting with 3 dense blocks, ending with 4 dense blocks (8) Trials with Grid Search with cross validation (9) Channel wise normalization of input data (later discarded and not used for final submission) |

**Sec 1 - Background and Task**: The assignment asks EIP2 participants to use a densenet and, using image augmentation with a SGD optimizer, obtain a 92% or more accuracy on the CIFAR 10 dataset. To this end a base of Code is provided and participants are expected to tune the hyperparameters and make ncessary model changes to reach the goal.

**Sec 2 - Comparison between Code provided and original Dense-Net**: The code provided to us participants implements the Densenet architecture, however, there are small differences:

1. In the original paper, the authors mention that they used the Densenet architecture with k=12 and depth of 40, totaling 1 million parameters and they report accuracy of 94.76% using data augmentation on the CIFAR 10 dataset. In the code provided, however, depth is taken as 32 and parameters are around 761K.
2. In the original paper, the authors hint at using the "he" kernel initialization scheme, which the originator, Kaiming He, claims to be superior to "Xavier" when training very deep networks. The code provided does not use a kernel initializer explicitly.
3. In the original paper, the authors use a channelwise normalization of the input data as a pre-processing step, while the code provided only rescales the data by dividing by 255 so that images can be plotted and seen humanly.
4. The authors of Densenet have conceptualized the architecture as Initial Convolution (3x3)  --> Series of Batch Normalization (BN) + ReLU + 3x3 Convolution --> Transition layer consisting of BN + 1x1 Convolution + 2x2 Average Pooling --> Output using Pooling followed by a Linear dense layer. In the code provided, we observe an additional ReLU activation in the Transition and Output.
5. The original paper mentions a learning rate schedule implementing a step decay with initial high learning rate of 0.1 reduced by a factor of 10 at 50% and 75% of the total number of 300 epochs for CIFAR 10. In the assignment we are given an upper limit of 250 epochs and the code provided does not include a learning rate schedule.

**Sec 3 - Initial attempts:** Initial attempts were with a 3 dense block architecture using a data augmentation scheme of horizontal flips and shifts. My experience was that these attempts always ended in a minimum of around 88.5% and once in that situation, no amount of changes to the learning rate would improve the situation; in fact with a 3 dense block architecture, even the training accuracy hits a plateau of around 91.3% suggesting that the model is not large enough to achieve the task at hand. At this point a decision was taken to enlarge the model to the maximum limit of 1 Million parameters and a hyper-parameter search was performed for a 4 dense blocks model in order to find the rough values of dropout along with a kernel constraint with which to run the model. The the output from the search is below:

***Sec 3.1 - Hyper-parameter search using sample data:***

`Best: 0.900300 using {'dropout_rate': 0.4, 'kernel_constraint': 3, 'learn_rate': 0.05}`
`0.900000 (0.000000) with: {'dropout_rate': 0.2, 'kernel_constraint': 2, 'learn_rate': 0.05}`
`0.900000 (0.000000) with: {'dropout_rate': 0.2, 'kernel_constraint': 2, 'learn_rate': 0.1}`
`0.900000 (0.000000) with: {'dropout_rate': 0.2, 'kernel_constraint': 3, 'learn_rate': 0.05}`
`0.900000 (0.000000) with: {'dropout_rate': 0.2, 'kernel_constraint': 3, 'learn_rate': 0.1}`
`0.900000 (0.000000) with: {'dropout_rate': 0.4, 'kernel_constraint': 2, 'learn_rate': 0.05}`
`0.900000 (0.000000) with: {'dropout_rate': 0.4, 'kernel_constraint': 2, 'learn_rate': 0.1}`
`0.900300 (0.000300) with: {'dropout_rate': 0.4, 'kernel_constraint': 3, 'learn_rate': 0.05}`
`0.900000 (0.000000) with: {'dropout_rate': 0.4, 'kernel_constraint': 3, 'learn_rate': 0.1}`

A few runs were performed to check if these values yielded good results and it was found that a constraint value of max_norm = 2 with a relatively small batch size = 64 was possibly a good choice for running the model. This batch size was chosen after experimentation, based on parts of surveyed literature that suggests that smaller batch sizes have neutral to positive effects on final model accuracy and help in managing GPU memory. Initial runs with augmentation and a Learning rate scheduler reducing learning rate based on the original paper was implemented. This did not lead to much improvement over the boundary of 88.5% that had already been reached with a shallow model.

***Sec 3.2 - Literature Survey for easier methods of tuning:*** A survey was done for other implementations on the Internet and for finding a quicker way than parameter search for tuning the model. Two important points immediately stood out:

**a.** There exists a possibility of multiple local minima in cases such as CIFAR-10 and once a local minima is achieved, it might be difficult to progress to the global minima. The loss surface, with skip connections as in the case of Resnet and Densenet, can be visualized as follows:

![https://www.cs.umd.edu/~tomg/img/landscapes/shortHighRes.png](https://www.cs.umd.edu/~tomg/img/landscapes/shortHighRes.png)  

**b.** A report on methods to tune hyper parameters without a full scale parameter search, called "A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay" can be found at https://arxiv.org/abs/1803.09820  Reference was made to this report to come up with possible values for hyper-parameters and the following decisions were taken

**Sec 4 - Decisions related to hyper parameters**

1. Use a small batch size of 64
2. Train the model initially on augmented training data to reduce variance between training and test accuracy and later use the original data for the last 50 epochs with a Reduce LR on plateau scheme for managing learning rate while the model makes its final run.
3. Use SGD optimizer with a momentum of 0.9 setting nesterov=True as used by the authors of Dense-net
4. Split the training into parts, start with high learning rate of 0.1 to accelerate training and then reduce the learning rate by a factor of 10 after 100 epochs and 150 epochs each.
5. Use 'he_uniform' or 'he_normal' kernel initialization as suggested by the authors of the Dense Net paper.
6. Use a kernel constraint max_norm=2 to prevent weights from exploding at relatively high learning rates used for initial training.

**Sec 5 - Model Training; observations and lessons learnt:**

i.) Initially a model was created with data augmentation (horizontal flip and shift by 10%) and run with a max_norm constraint to prevent weights from growing too large at high learning rates. After 200 epochs, we observed that the capacity of the model was limited due to kernel constraint with the maximum training accuracy observed as 93.1%, validation accuracy was around 89%.

* Decision (5) and (6) of Section 4 above were later changed to *'he_uniform'* and a *l2 regularization of 1e(-4)* as a kernel constraint was found the throttle the capacity of the model.

ii) It was observed that the time taken to train the model from scratch is of the order of 8 hours, this hampers experimentation and indicates the need for quicker means of training, either through use of higher end hardware or transfer learning.

iii) Channel wise normalization did not seem to affect the final accuracy by much, especially if a rescaling by division by 255 had already been carried out. Final model was run without channel wise normalization.

iv) Using more steps per epoch did not seem to significantly affect the final outcome; literature survey seems to suggest that this happens because we are using Batch Normalization, drop out and other means of controlling the neural network. The final run was therefore with standard steps per epoch = integer(sample size/ batch size).

v) An examination of the trend of training loss, training accuracy and test loss, test accuracy indicates that further improvement in accuracy is possible, using the same model, with suitable fine tuning. The values of training accuracy, pattern of training loss and the generalization error [test/ validation loss - training loss] suggest that:

* Further training accuracy improvement may be possible by improving on the combination of hyper-parameters chosen; final training accuracy observed is 0.9495; literature suggests that this could reach the 0.9800 - 0.9900 range.
* The variance error is observed to be of the *4% range*, this again suggests that regularization used could be further tuned to bring this down to the *1% range* thereby increasing test accuracy. The best test accuracy observed is 0.9100 which is 1% short of the assignment target of 92%.

vi) Infrastructure constraints such as Internet connectivity cause multiple disconnections and ways and means to circumvent the effects - viz. reloading model with checkpoint - are necessary.

vii) Last, but probably very important, ***one must have patience, time and tenacity*** to see the training through to its end without giving up midway.