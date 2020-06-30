

import numpy as np
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, LSTM
from keras.layers import TimeDistributed, Masking
from keras import optimizers
from keras import regularizers
from sklearn.metrics import accuracy_score
import os
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels
from Save_Output import save_output


def LSTM_model_veracity(x_train, y_train, x_test):
    #Neural Network model created and trained in GOOGLE COLAB and the model is saved and loaded here locally.
    """
    model = Sequential()
    num_features = x_train.shape[2]
    model.add(Masking(mask_value=0., input_shape=(None, num_features)))
    model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
    model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2,return_sequences=False))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax',activity_regularizer=regularizers.l2(3e-4)))
    adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, y_train,epochs=100, shuffle=True, class_weight=None)
    model_json = model.to_json()
    with open("Saved_NNModel/model_sdqc.json", "w") as json_file:
      json_file.write(model_json)
    model.save_weights("Saved_NNModel/model_sdqc.h5")
    print("Saved model to disk")
    """
    json_file = open('Saved_NNModel/model_sdqc.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("Saved_NNModel/model_sdqc.h5")
    print("Created LSTM Model")
    pred_probabilities = model.predict(x_test)
    confidence = np.max(pred_probabilities, axis=1)
    Y_pred = model.predict_classes(x_test)
    return Y_pred, confidence

def main():
    x_train = np.load(os.path.join('Feature_Engineering/Saved_dataset_sdqc','train/train_array.npy'))
    y_train = np.load(os.path.join('Feature_Engineering/Saved_dataset_sdqc', 'train/labels.npy'))
    y_train = to_categorical(y_train, num_classes=None)
    x_test = np.load(os.path.join('Feature_Engineering/Saved_dataset_sdqc', 'dev/train_array.npy'))
    y_test = np.load(os.path.join('Feature_Engineering/Saved_dataset_sdqc', 'dev/labels.npy'))
    ids_test = np.load(os.path.join('Feature_Engineering/Saved_dataset_sdqc','dev/ids.npy'))
    y_pred, confidence = LSTM_model_veracity(x_train, y_train, x_test)
    trees, tree_prediction, tree_label,veracity_confidence = branch2treelabels(ids_test, y_test,y_pred,confidence)
    for i in range(len(veracity_confidence)):
        if tree_prediction[i]==2:
            veracity_confidence[i]=0
            tree_prediction[i]=1
        if tree_label[i]==2:
            tree_label[i]=1
    dummy=[]
    save_output(dummy,dummy,trees,tree_prediction,veracity_confidence)
    #print ("Charliehedbo event is the Test Data")
    print (trees)
    print ("Predictions",(tree_prediction))
    print ("Labels",(tree_label))
    print ("Confidence",(veracity_confidence))
        
    print ("Accuracy :",(accuracy_score(tree_label,tree_prediction)))
    from sklearn.metrics import classification_report
    print(classification_report(tree_label,tree_prediction,target_names=["true","false"]))

    
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    skplt.metrics.plot_confusion_matrix(tree_label,tree_prediction)
    plt.show()



main()
