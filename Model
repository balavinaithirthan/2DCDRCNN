from google.colab import drive
drive.mount('/content/drive')

base_folder = "./sample_data/"
base_project_folder = base_folder + "project/"

# copy the zip file to the sample_data folder
!cp "drive/My Drive/2DCDRCNN/projectdata.zip" ./sample_data/
!cp "drive/My Drive/2DCDRCNN/R2.py" .

from zipfile import ZipFile 
  
# specifying the zip file name 
file_name = base_folder + "projectdata.zip"
  
# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall() 
    print('Done!') 
    
    import pandas as pd
# load the drug and gene expression data set 
drug_gene_expression_dataset = pd.read_csv("sample_data/project/data.csv")

import numpy as np

# load onehot enencoding of all drugs
onehot_encoded_drug_list = {}
# load the normailized gene expression of all drugs
normailized_gene_expression_list = {}
for item in drug_gene_expression_dataset.iterrows() :
  drug_name = item[1]['drug names']
  cell_name = item[1]['cell names']
  if (drug_name not in onehot_encoded_drug_list) :
    onehot_encoded_drug = pd.read_csv(drug_name, header=None, dtype=np.float64).to_numpy()
    onehot_encoded_drug_list[drug_name] =  np.reshape(onehot_encoded_drug, [188,28])
  if (cell_name not in normailized_gene_expression_list) :
    normailized_gene_expression = pd.read_csv(cell_name, header=None, dtype=np.float64).to_numpy()
    normailized_gene_expression_list[cell_name] = normailized_gene_expression.flatten()
print(len(normailized_gene_expression_list))
print(len(onehot_encoded_drug_list))

#create a feature list consisting of onehot drug representation, normailized gene expression and ic50 target value
onehot_encoded_drug_feature_list = []
normailized_gene_expression_feature_list = []
ic50_target_list = []
for item in drug_gene_expression_dataset.iterrows() :
  drug_name = item[1]['drug names']
  cell_name = item[1]['cell names']
  ic50s = item[1]['ic50']
  normailized_gene_expression_feature_list.append(normailized_gene_expression_list[cell_name])
  onehot_encoded_drug_feature_list.append(onehot_encoded_drug_list[drug_name])
  ic50_target_list.append(ic50s)
onehot_encoded_drug_feature = np.stack( onehot_encoded_drug_feature_list, axis=0 )
normailized_gene_expression_feature = np.stack( normailized_gene_expression_feature_list, axis=0 )
ic50s = np.asarray(ic50_target_list)
print(onehot_encoded_drug_feature.shape)
print(normailized_gene_expression_feature.shape)
print(ic50s.shape)

from keras.models import Sequential, Model, Input
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Concatenate, Dropout
import tensorflow as tf
import numpy as np
import R2

onehot_encoded_drug_input = Input(shape=(188,28), dtype=tf.float32, name="onehot_encoded_drug_input")
d = Conv1D(filters=40, kernel_size=7, padding="same", activation="relu", use_bias=True, name="drug_cnn_layer_1")(onehot_encoded_drug_input)
d = MaxPooling1D(pool_size=3, strides=3, padding="same", name="drug_cnn_layer_2")(d)
d = Conv1D(  filters=80, kernel_size=7, padding="same", activation="relu", use_bias=True, name="drug_cnn_layer_3")(d)
d = MaxPooling1D(pool_size=3, strides=3, padding="same", name="drug_cnn_layer_4")(d)
d = Conv1D(  filters=60, kernel_size=7, padding="same", activation="relu", use_bias=True, name="drug_cnn_layer_5")(d)
d = MaxPooling1D(pool_size=3, strides=3, padding="same", name="drug_cnn_layer_6")(d)

normailized_gene_expression_input = Input(shape=(2500,1), name="normailized_gene_expression_input")
g = Conv1D(filters=40, kernel_size=7, padding="same", activation="relu", use_bias=True, name="gene_expression_cnn_layer_1")(normailized_gene_expression_input)
g = MaxPooling1D(pool_size=3, strides=3, padding="same", name="gene_expression_cnn_layer_2")(g)
g = Conv1D(  filters=80, kernel_size=7, padding="same", activation="relu", use_bias=True, name="gene_expression_cnn_layer_3")(g)
g = MaxPooling1D(pool_size=3, strides=3, padding="same", name="gene_expression_cnn_layer_4")(g)
g = Conv1D(  filters=60, kernel_size=7, padding="same", activation="relu", use_bias=True, name="gene_expression_cnn_layer_5")(g)
g = MaxPooling1D(pool_size=3, strides=3, padding="same", name="gene_expression_cnn_layer_6")(g)

fcn = Concatenate(1, name="combine_drug_and_gene_expression_cnns")([d, g])
fcn = Flatten(name="fcn_layer_1")(fcn)
fcn = Dense(1024, activation='relu',use_bias=True, name="fcn_layer_2")(fcn)
fcn = Dropout(0.2, name="fcn_layer_3")(fcn)
fcn = Dense(1024, activation='relu',use_bias=True, name="fcn_layer_4")(fcn)
fcn = Dropout(0.2, name="fcn_layer_5")(fcn)
fcn_output = Dense(1,  activation='sigmoid',use_bias=True, name="fcn_output_layer_6")(fcn)

model = Model( name="2d_cdr_cnn",  
    inputs=([onehot_encoded_drug_input,normailized_gene_expression_input]),
    outputs=fcn_output
)


model.compile( optimizer='adam',
               loss="mean_squared_error",
               metrics=[tf.keras.metrics.RootMeanSquaredError(), R2.RSquare(), 'mae'])

model.summary()
