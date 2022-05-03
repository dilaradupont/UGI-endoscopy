import keras
from keras.utils import np_utils
from keras.layers import Dense, Flatten 
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sbs

IMAGE_SIZE = [224, 224]
TRAINING_PATH = '/home/dilara/UGI-endoscopy/training-data'
TESTING_PATH = '/home/dilara/UGI-endoscopy/testing-data'
CLASSES = ['other1', 'pylorus', 'z-line', 'retroflex-stomach']
TEST_CLASSES = ['other2', 'pylorus', 'z-line', 'retroflex-stomach']

def get_paths(path, class_list):
    '''
    Get all the image paths and store them in list.
    
    Args:
        path (str): path to training or testing data
        class_list (list): list of classes/landmarks
    Returns:
        list: list of lists of all the image paths for each class
    '''
    all_paths = []
    for landmark in class_list:
        other_path = os.path.join(path, landmark, '*')
        all_paths.append(sorted(glob.glob(other_path)))
    return all_paths

def get_class_num(path):
    '''
    Find and return the number of images per class.
    
    Args:
        path (list): list of lists of all the image paths for each class
    Returns:
        list: number of images per class
    '''
    len_lis = []
    for lis in path:
        x = 0
        for item in lis:
            x+=1
        len_lis.append(x)
    print(len_lis)
    return len_lis

def create_labels(len_lis, class_list):
    '''
    Create a list with the image labels, where other:0, pylorus:1,
     z-line:2, retroflex-stomach:3.
    
    Args:
        len_lis (list): number of images per class
        class_list (list): list of the labels/classes 
    Returns:
        list: all the image labels
    '''
    labels_lis = []
    for i in range(len(class_list)):
        labels_lis.append([i] * len_lis[i])
        all_labels = list(np.concatenate(labels_lis).flat)
        all_labels = np.array(all_labels)
    return all_labels

def get_pix(path_lis):
    '''
    Extract the image data and store in list. 
    
    Args:
        path_lis (list): list of image paths
    Returns:
        list: list of image data
    '''
    pix = []
    for path in path_lis:
        image = load_img(path, color_mode='rgb', target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)
        pix.append(image)
    return pix

def plot_model_history(model_history):
    ''' 
    Plot the model's training and validation history.
    
    Args:
        model_history (History object): Trained model's history
    '''
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(0,len(model_history.history['accuracy'])),\
        model_history.history['accuracy'])
    axs[0].plot(range(0,len(model_history.history['val_accuracy'])),\
        model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Training Accuracy', 'Validation Accuracy'], loc='best')

    axs[1].plot(range(0,len(model_history.history['loss'])),\
        model_history.history['loss'])
    axs[1].plot(range(0, len(model_history.history['val_loss'])),\
        model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Training Loss', 'Validation Loss'], loc='best')

    plt.show()

def generate_cm(label_true, label_predict, norm=True):
    ''' Generate the confusion matrix of the model's predictions. The confusion
    Matrix can be normalised or non-normalised.
    
    Source for confusion matrix normalisation:     
    https://stackoverflow.com/questions/59049746/limit-normalised-confusion-matrix-to-2-decimal-points
    
    Args:
    label_true (array): array of true labels
    label_predict (array): array of most probable label predicted by model
    norm (bool): normalise the confusion matrix (True)
    '''
    cm = confusion_matrix(label_true, label_predict)
    if norm == True:
        cm = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], \
            decimals=2)
    plt.figure(figsize=(10,10))
    ax = plt.subplot()
    sbs.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g")
    sbs.set(font_scale=2) 

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    title_font = {'size':'36'}  
    ax.set_title('Confusion Matrix', fontdict=title_font)
    ax.tick_params(axis='both', which='major', labelsize=20) 
    ax.xaxis.set_ticklabels(['0', '1', '2', '3'])
    ax.yaxis.set_ticklabels(['0', '1', '2', '3'])
    plt.show()

def main():
    all_paths_train = get_paths(TRAINING_PATH, CLASSES)
    all_paths_test = get_paths(TESTING_PATH, TEST_CLASSES)
    len_lis_train = get_class_num(all_paths_train)
    len_lis_test = get_class_num(all_paths_test)

    all_paths_train = list(np.concatenate(all_paths_train).flat)
    all_paths_test = list(np.concatenate(all_paths_test).flat)
    all_labels_train = create_labels(len_lis_train, CLASSES)
    all_labels_test = create_labels(len_lis_test, TEST_CLASSES)
    print(len(all_labels_train))

    pix_train = get_pix(all_paths_train)
    pix_test = get_pix(all_paths_test)
    pix_train = np.array(pix_train)
    pix_test = np.array(pix_test)

    pix_train, pix_val, label_train, label_val = train_test_split(pix_train,
     all_labels_train, train_size=0.75, random_state = 42)

    # Storing labels in one hot encoded format
    label_train = keras.utils.np_utils.to_categorical(label_train,
     num_classes=len(CLASSES))
    label_val = keras.utils.np_utils.to_categorical(label_val,
     num_classes=len(CLASSES))
    label_test = keras.utils.np_utils.to_categorical(all_labels_test,
     num_classes=len(TEST_CLASSES))

    vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    x = Flatten()(vgg.output)
    prediction = Dense(len(CLASSES), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    model.summary()

    history = model.fit(pix_train, label_train, batch_size=64, epochs=10,
     validation_data=(pix_val, label_val))
    plot_model_history(history)

    # Evaluation of Performance with Validation Dataset
    label_predict = np.argmax(model.predict(pix_val), axis=1)
    label_true = np.argmax(label_val, axis=1) 
    classes = ['other (0)', 'pylorus (1)', 'z-line (2)','retroflex-stomach (3)']
    print("Classification report:\n", classification_report(label_true,
     label_predict, target_names=classes))
    generate_cm(label_true, label_predict, False)
    generate_cm(label_true, label_predict)

    # Evaluation of Performance with Testing Dataset
    label_predict = np.argmax(model.predict(pix_test), axis=1)
    label_true = np.argmax(label_test, axis=1) 
    print("Classification report:\n", classification_report(label_true,
     label_predict, target_names=classes))
    generate_cm(label_true, label_predict, False)
    generate_cm(label_true, label_predict)

if __name__ == '__main__':
    main()

