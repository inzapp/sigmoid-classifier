# Sigmoid Classifier

The sigmoid classifier is a classifier that sigmoid output as activation function.

A typical classification model has a softmax avtivation in the last layer and is trained through the CCE loss function.

It's very good and we're using it a lot.

So why use the sigmoid activation when there is a very good combination of softmax + CCE?

## Unknown class

Most classification problems do not have an unknown class.

An unknown class is a class that does not correspond to any of the classes you want to classify.

It is also called no_obj or not_obj.

We found that classifiers using sigmoid showed better performance in classification problems with unknown classes.

The training label on the sigmoid classifier does not contain classes that distinguish between unknown classes.

However, it trains all classes by giving them zero.

Let me give you an example.

The label for the softmax classifier is one-hot-vector, which gives 1 to the index pointing to that class and 0 to none.

class_2 (of 3 classes) = [0, 0, 1]

If you are adding an unknown class in this state, the label is as follows:

unknown (of 3 classes) = [0, 0, 0, 1]

On the other hand, the sigmoid classifier label method is as follows:

class_2 (of 3 classes) = [0, 0, 1]

unknown (of 3 classes) = [0, 0, 0]

Softmax cannot be used because there exists a label that does not contain 1.

And it is replaced by sigmoid.

## Why Sigmoid?

The following are the reasons for using sigmoid activation in the classification model.

### 1. sigmoid classifier can improve the accuracy of the model by additionally training special labels with all nodes zero, which cannot be expressed in softmax.

It is common to use softmax in classification models that are usually trained as one not vector.

However, We improve classification accuracy by training information about unknown classes together in the classification model.

And the value of all nodes in the label for an unknown class is zero and cannot be expressed as one hot vector.

Therefore, training is not possible with softmax.

### 2. Flexibility for Label smoothing

If you use softmax, the sum of label tensors is forced to 1.

Therefore, when training with label smoothing, the range of labels that can be trained is constrained.

For example, if you want to learn a classification model with a tensor of [0.1, 0.1, 0.9], you can't express it with softmax because the sum of these tensors is 1.

However, sigmoid allows us to test a variety of possibilities because it increases the range of labels that can be learned from this perspective.

[0.4, 0.4, 0.9] You can also train with fully customized labels in this form.

### 3. Independent logistic regression(multi classification)

I am learning the model by replacing the classification problem with an independent logistic regression problem instead of a simple multiple-choice problem.

This enables independent probability inference for each class.

This enables multi-classification.

This maximizes the effectiveness when data for the unknown class mentioned in No. 1 is included, compared to not.

### 4. Zero centered logit

With softmax, the distribution of logit values before going through softmax cannot be predicted without using the normalization layer.

This indirectly indicates that the stability of softmax is related to the distribution of logits.

However, using sigmoid limits the logit to between -20 and 20.

In deep learning training, zero centered logit has the effect of stabilizing the training by itself.

## Perspective of probability

The output of a classifier using softmax is often interpreted as the probability that it is a corresponding class.

But I always doubted it. Does this really mean anything as a probability?

Suppose you have a softmax classification model that classifies five classes.

And because of the last softmax layer, the sum of the output values for this model is 1.

<img src="/md/softmax.jpg" width="400"><br>

Because of exp, the larger value of the output will receive more weight and the smaller value will be smaller.

This is strange.

It's as if they're manipulating the value to make it seem more certain.

On the other hand, I think the values that the models trained with sigmoid and BCE output are reliable from a probabilistic perspective.

This is because the probability of being that class for each class is the same as being a logistic regression.

Am I wrong?

I have made and tested many kinds of classification models.

The classification model using softmax seemed to be better trained.

This is because train acc and val acc were more stable during training.

However, when testing actual unseened data,

The classification model using sigmoid gave me a better result.

And the output of the model seemed more reliable from a probabilistic point of view.

Am I missing something?

Anyone can share your opinion on this.
