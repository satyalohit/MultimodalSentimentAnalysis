
import numpy as np
np.random.seed(1337) 

from keras import backend as K
from keras.layers.core import Lambda
from tensorflow.keras.layers import Layer

from keras.models import Sequential
from keras.layers import Input,Dense,GRU,LSTM,Concatenate,Dropout,Activation,Add, Masking
from keras.layers.pooling import AveragePooling1D,MaxPooling1D
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Reshape
from keras.backend import shape
from keras.utils.vis_utils import plot_model
# from keras.utils import plot_model
from keras.layers.merge import Multiply,Concatenate
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import RMSprop,Adadelta,Adam

from keras.callbacks import Callback

import sys

def createOneHot(train_label,  test_label):


  maxlen = int(max(train_label.max(), test_label.max()))
  
  train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen+1))
  test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen+1))
  
  for i in range(train_label.shape[0]):
    for j in range(train_label.shape[1]):
      train[i,j,train_label[i,j]]=1

  for i in range(test_label.shape[0]):
    for j in range(test_label.shape[1]):
      test[i,j,test_label[i,j]]=1

  return train,  test

def calc_test_result(result, test_label, test_mask):

  true_label=[]
  predicted_label=[]

  for i in range(result.shape[0]):
    for j in range(result.shape[1]):
      if test_mask[i,j]==1:
        true_label.append(np.argmax(test_label[i,j] ))
        predicted_label.append(np.argmax(result[i,j] ))
    
  print("Confusion Matrix :")
  print(confusion_matrix(true_label, predicted_label))
  print("Classification Report :")
  print(classification_report(true_label, predicted_label,digits=4))
  print("Accuracy ", accuracy_score(true_label, predicted_label))
  print("Macro Classification Report :")
  print(precision_recall_fscore_support(true_label, predicted_label,average='macro'))
  print("Weighted Classification Report :")
  print(precision_recall_fscore_support(true_label, predicted_label,average='weighted'))
 








def crop(a):
    return a[:, 0:text_dim]
def crop1(a):
    return a[:, text_dim:(text_dim+visual_dim)]
def crop2(a):
    return a[:, (text_dim+visual_dim):dim]
def output_of_lambda(input_shape):
    return (input_shape[0], text_dim)
def output_of_lambda1(input_shape):
    return (input_shape[0], visual_dim)
def output_of_lambda2(input_shape):
    return (input_shape[0], audio_dim)










def load_bimodal_activations():     
  with open('./bimodal.pickle', 'rb') as handle:
      activations = pickle.load(handle, encoding = 'latin1')  
  merged_train_data = activations['train_bimodal']
  merged_test_data = activations['test_bimodal']
  train_mask=activations['train_mask']
  test_mask=activations['test_mask']
  train_label=activations['train_label']
  test_label=activations['test_label']


  
  for i in range(activations['train_bimodal'].shape[0]):
    for j in range(activations['train_bimodal'].shape[1]):
      if train_mask[i][j] == 0.0 :

        merged_train_data[i,j,:]=0.0

  for i in range(activations['test_bimodal'].shape[0]):
    for j in range(activations['test_bimodal'].shape[1]):
      if test_mask[i][j] == 0.0 :

        merged_test_data[i,j,:]=0.0

  audio_dim=500
  visual_dim=500
  text_dim=500
  dim = audio_dim + visual_dim + text_dim
  max_len = merged_train_data.shape[1]
  dim_proj=450

  return merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask, max_len


def load_unimodal_activations():

  with open('./unimodal.pickle', 'rb') as handle:
      unimodal_activations = pickle.load(handle, encoding = 'latin1')

  merged_train_data = np.concatenate((unimodal_activations['text_train'], unimodal_activations['audio_train'], unimodal_activations['video_train']), axis=2)
  merged_test_data = np.concatenate((unimodal_activations['text_test'], unimodal_activations['audio_test'], unimodal_activations['video_test']), axis=2)
  
  train_mask=unimodal_activations['train_mask']
  test_mask=unimodal_activations['test_mask']
  train_label=unimodal_activations['train_label']
  test_label=unimodal_activations['test_label']


  
  for i in range(unimodal_activations['audio_train'].shape[0]):
    for j in range(unimodal_activations['audio_train'].shape[1]):
      if train_mask[i][j] == 0.0 :

        merged_train_data[i,j,:]=0.0

  for i in range(unimodal_activations['audio_test'].shape[0]):
    for j in range(unimodal_activations['audio_test'].shape[1]):
      if test_mask[i][j] == 0.0 :

        merged_test_data[i,j,:]=0.0

  
  global audio_dim, visual_dim, text_dim, dim, max_len, dim_proj


  audio_dim=unimodal_activations['audio_train'].shape[2]
  visual_dim=unimodal_activations['video_train'].shape[2]
  text_dim=unimodal_activations['text_train'].shape[2]
  dim = audio_dim + visual_dim + text_dim
  max_len = merged_train_data.shape[1] 
  dim_proj=450

  return merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask


def Bimodal():

  class hadamard2(Layer):
      def __init__(self, prefix, **kwargs):
          self.supports_masking = True
          self.prefix = prefix
          super(hadamard2, self).__init__(**kwargs)

      def build(self, input_shape):
          
          self.output_dim = dim_proj

          self.kernel1 = self.add_weight(name=self.prefix+'kernel1', 
                                        shape=(self.output_dim,),
                                        initializer='TruncatedNormal',
                                        trainable=True)
          self.kernel2 = self.add_weight(name=self.prefix+'kernel2', 
                                        shape=(self.output_dim,),
                                        initializer='TruncatedNormal',
                                        trainable=True)
          self.bias = self.add_weight(name=self.prefix+'bias', 
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)
          super(hadamard2, self).build(input_shape)  

      def call(self, x):
          assert (K.int_shape(x[0])[1] <= dim_proj)
          return K.tanh(x[0]*self.kernel1 + x[1]*self.kernel2 + self.bias)

      def compute_output_shape(self, input_shape):
          return (input_shape[0][0],self.output_dim)



  merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask = load_unimodal_activations()
  
  

  x=Input(shape=(audio_dim+visual_dim+text_dim,), name='bi_input')
  masked = Masking(mask_value =0.0 , name='bi_mask')(x)
  x1 = Lambda(crop,output_shape=output_of_lambda, name='bi_crop1')(masked)
  x2 =Lambda(crop1,output_shape=output_of_lambda1, name='bi_crop2')(masked)
  x3 =Lambda(crop2,output_shape=output_of_lambda2, name='bi_crop3')(masked)

  dense1=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense1')(x1)
  dense2=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense2')(x2)
  dense3=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense3')(x3)
  fused_1_2=hadamard2('1')([dense1,dense2])
  fused_1_3=hadamard2('2')([dense1,dense3])
  fused_2_3=hadamard2('3')([dense2,dense3])
  bimodal_1_2=Model(x,outputs=fused_1_2)
  bimodal_1_3=Model(x,outputs=fused_1_3)
  bimodal_2_3=Model(x,outputs=fused_2_3)


  

  input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input1')
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm1')(input_data)
  inter = Dropout(0.9, name='bi_dropout11')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis11'))(inter)
  output1 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis12'))(inter)
  lstm1 = Model(input_data, output1)
  aux1 = Model(input_data, inter)


  input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input2')
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm2')(input_data)
  inter = Dropout(0.9, name='bi_dropout21')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis21'))(inter)
  output2 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis22'))(inter)
  lstm2 = Model(input_data, output2)
  aux2 = Model(input_data, inter)


  input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input3')
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm3')(input_data)
  inter = Dropout(0.9, name='bi_dropout31')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis31'))(inter)
  output3 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis32'))(inter)
  lstm3 = Model(input_data, output3)
  aux3 = Model(input_data, inter)

   

  main_input=Input(shape=(max_len,audio_dim+visual_dim+text_dim,), name='bi_main_input')

  video_uttr_1_2=TimeDistributed(bimodal_1_2, name='bi_tdis_b12')(main_input)
  video_uttr_1_3=TimeDistributed(bimodal_1_3, name='bi_tdis_b13')(main_input)
  video_uttr_2_3=TimeDistributed(bimodal_2_3, name='bi_tdis_b23')(main_input)
  
  aux_1_2=aux1(video_uttr_1_2)
  aux_1_3=aux2(video_uttr_1_3)
  aux_2_3=aux3(video_uttr_2_3)



  context_1_2=lstm1(video_uttr_1_2)
  BM1 = Model(main_input, context_1_2)
  Aux1 = Model(main_input, aux_1_2)
  context_1_3=lstm2(video_uttr_1_3)
  BM2 = Model(main_input, context_1_3)
  Aux2 = Model(main_input, aux_1_3)
  context_2_3=lstm3(video_uttr_2_3)
  BM3 = Model(main_input, context_2_3)
  Aux3 = Model(main_input, aux_2_3)


 

  optimizer=Adam(lr=0.001)
  BM1.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM1.fit(merged_train_data, train_label,
                  epochs=200,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)
  
  train_result1 = Aux1.predict(merged_train_data)
  test_result1 = Aux1.predict(merged_test_data)

  BM2.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM2.fit(merged_train_data, train_label,
                  epochs=200,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)
  train_result2 = Aux2.predict(merged_train_data)
  test_result2 = Aux2.predict(merged_test_data)

  BM3.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM3.fit(merged_train_data, train_label,
                  epochs=200,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)
  train_result3 = Aux3.predict(merged_train_data)
  test_result3 = Aux3.predict(merged_test_data)

  train_bimodal = np.concatenate((train_result1, train_result2, train_result3), axis=2)
  test_bimodal = np.concatenate((test_result1, test_result2, test_result3), axis=2)

  bimodal_activations={}
  bimodal_activations['train_bimodal'] = train_bimodal
  bimodal_activations['test_bimodal'] = test_bimodal
  bimodal_activations['train_mask']=train_mask
  bimodal_activations['test_mask']= test_mask
  bimodal_activations['train_label']=train_label
  bimodal_activations['test_label']=test_label

  
  print("Saving bimodal activations")
  with open('bimodal.pickle', 'wb') as handle:
       pickle.dump(bimodal_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':


  Bimodal()

  







