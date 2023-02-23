# VGG11

VGG is a neural network model that uses convolutional neural network (CNN) layers and was designed for the ImageNet challenge, which it won in 2014.
VGG is not a single model, but a family of models that are all similar but have different configurations. Each configuration specifies the number of layers and the size of each layer. The configurations are listed in table 1 of the VGG paper and denoted by a letter, although recently they are just referred to as the number of layers with weights in the model, e.g. configuration "A" has 11 layers with weights so is known as VGG11.
![V1](https://user-images.githubusercontent.com/48100680/220989803-f8290e31-eb73-4bcb-98ff-0b6a7d8454f5.jpg)
 
Below is the architecture of configuration "D", also known as VGG16, for a 224x224 color image.
![V2](https://user-images.githubusercontent.com/48100680/220989942-fd3f7992-0ccd-44b5-b5f7-88eaded1a362.jpg)


The other commonly used VGG variants are VGG11, VGG13 and VGG19, which correspond to configurations "A", "B", and "E". Configurations "A-LRN" and "C" - which is the same as "D" but with smaller filter sizes in some convolutional layers - are rarely used.

As in the previous notebook, we will use the CIFAR10 dataset and the learning rate finder introduced here. However we will be making use of pre-trained models.

Usually we will initialize our weights randomly - following some weight initialization scheme - and then train our model. Using a pre-trained model means some - potentially all - of our model's weights are not initialized randomly, but instead taken from a copy of our model that has already been trained on some task. The task the model has been pre-trained on does not necessarily have to match the "downstream task" - the task we want to use the pre-trained model on. For example, a model that has been trained to classify images can then be used to detect objects within an image.

The theory is that these pre-trained models have already learned high level features within images that will be useful for our task. This means we don't have to learn them from scratch when using the pre-trained model for our task, causing our model to converge earlier. We can also think of the pre-trained model as being a very good set of weights to initialize our model from, and using pre-trained models usually leads to a performance improvement compared to initializing our weights randomly.

The act of using a pre-trained model is generally known as transfer learning, as we are learning to transfer knowledge from one task to another. It is also referred to as fine-tuning, as we fine-tune our parameters trained on one task to the new, downstream task. The terms transfer learning and fine-tuning are used interchangeably in machine learning.
