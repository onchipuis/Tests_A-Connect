### Modified by : Ricardo Vergel - October 10/2020
### Some useful functions for neural networks
#### plot_image and plot_value_array taken from https://www.tensorflow.org/tutorials/keras/classification

import matplotlib.pyplot as plt
import numpy as np


def plot_image(i, predictions_array, true_label, img,class_names): # Plot test image 
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("({}) {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[int(true_label)]),
                                color=color)

def plot_value_array(i, predictions_array, true_label): # Plot test label
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[int(true_label)].set_color('blue')

def plot_test_imgs(predictions_array,test_labels,test_images, class_names,rows, cols,*args): #Plot a batch of test images and labels
	path = "/home/rvergel/Desktop/Library_AConnect_TG/Tensorflow/Graphs/Predictions/"
	string = args[0]
	images = rows*cols
	plt.figure(figsize=(2*2*cols, 2*rows))
	for i in range(images):
  		plt.subplot(rows, 2*cols, 2*i+1)
  		plot_image(i, predictions_array[i], test_labels, test_images,class_names)
  		plt.subplot(rows, 2*cols, 2*i+2)
  		plot_value_array(i, predictions_array[i], test_labels)
	plt.tight_layout()
	if(args!=None):
		plt.savefig(path+string+".png")
	#plt.show()


def plot_train_imgs(train_images,train_labels,class_names,win_width,win_height,rows,cols): #Plot a batch of train images and labels
	plt.figure(figsize=(win_width,win_height))
	for i in range(len(class_names)):
		plt.subplot(rows,cols,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_images[i],cmap=plt.cm.binary)
		plt.xlabel(class_names[int(train_labels[i])])
	plt.show()

def plot_history(result,option): #Plot the loss or the accuracy of our nn vs epochs. option: 1 for loss 2 for accuracy
	if (option == 1):
		plt.plot(range(1, len(result) + 1), result, 'b', label = 'Training loss')
		plt.title('Training loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
	elif(option == 2):
		plt.plot(range(1, len(result) + 1), result, 'b', label = 'Training accuracy')
		plt.title('Training accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()

	else:
		print('Invalid option, type 1 for loss graph, 2 for accuracy graph')
	plt.show()

def plot_full_history(acc,val_acc,loss,val_loss,epochs_range,*args):
	path = "/home/rvergel/Desktop/Library_AConnect_TG/Tensorflow/Graphs/Training/"
	string = args[0]
	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	if(args!=None):
		plt.savefig(path+string+".png")
	#plt.show()
	
