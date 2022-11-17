#!/usr/bin/env python
# coding: utf-8

# ## Imports and Data Preprocessing
# 
# Import data and do basics of removing extraneous data

# In[3]:


# Initial imports
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import np_utils
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Import Verlander dataset 

verlander_df = pd.read_csv(Path("resources/verlander_update.csv"))

# display(verlander_df.head())


# In[5]:


# Clean dataset 

verlander_df = verlander_df.drop(columns=['des',
'at_bat_number', 
'inning', 
'zone', 
'player_name', 
'batter', 
'pitcher', 
'events',
'bb_type',
'hit_location',
'home_team',
'away_team'])

# display(verlander_df.head())


# In[6]:


# Encode 1st, 2nd and 3rd bases with 1s and 0s

# Fill NaN to 0 
verlander_df['on_3b'] = verlander_df['on_3b'].fillna(0)
verlander_df['on_2b'] = verlander_df['on_2b'].fillna(0)
verlander_df['on_1b'] = verlander_df['on_1b'].fillna(0)

# Change batter IDs to 1 
verlander_df['on_3b'][verlander_df['on_3b'] > 0.0] = 1.0
verlander_df['on_2b'][verlander_df['on_2b'] > 0.0] = 1.0
verlander_df['on_1b'][verlander_df['on_1b'] > 0.0] = 1.0

# display(verlander_df.head())


# In[7]:


# Fill NaN in description column

verlander_df['description'] = verlander_df['description'].fillna('nothing')


# ## Feature Engineering

# In[8]:


# Shift pitches so that model does not know the upcoming pitch 

verlander_df['pitch_name'] = verlander_df['pitch_name'].shift(-1).dropna()
verlander_df['type'] = verlander_df['type'].shift(-1).dropna()

# display(verlander_df)


# In[9]:


# Converting batting score and fielding score to one column.
# Positive number means fielding team is winning and negative number means batting team is winning. 
verlander_df['score_diff'] = verlander_df['fld_score'] - verlander_df['bat_score']


# Drop batting score and fielding score columns now that you have the score differential
verlander_df.drop(columns = ['bat_score', 'fld_score'], inplace=True)


# In[10]:


# Feature engineering to count the number of pitches JV has thrown each outing

verlander_df['ones'] = 1
pitch_count_df = verlander_df[['game_date', 'ones']]
pitch_count_df['pitch_count'] = pitch_count_df.groupby(['game_date']).cumcount(ascending = False)
pitch_count_df['pitch_count'] = pitch_count_df['pitch_count'] + 1

verlander_df = pd.concat([verlander_df, pitch_count_df['pitch_count']], join='inner', axis=1)


# In[11]:


# Feature engineering to change the ball and strike count into one column as a string

verlander_df['count'] = verlander_df['balls'].astype(str) +'-'+ verlander_df['strikes'].astype(str)
verlander_df.drop(columns=['balls', 'strikes'], inplace=True)
# verlander_df.head()


# ## Data Processing 
# 
# Prepare data to be fed into model

# In[12]:


# Split into X and y 

X = verlander_df.drop(columns=['pitch_type', 'game_date'])
y= verlander_df['pitch_type']

# display(y.value_counts)


# In[13]:


# Use get_dummies to encode categorical variables 

X = pd.get_dummies(X)

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

# display(X.head())
# display(dummy_y)

# display(X.columns)

# SL = index 3
# CH = index 0
# FF = index 2
# CU = index 1


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, random_state=1)

# X_test.head()


# ## Deep Learning/Neural Network Architecture
# 
# Using the method from the module, the following basis is used to design the first iteration of the neural network: 
# 
# *the mean of the number of input features and the number of neurons in the output layer ((number of input features + number of neurons in output layer) / 2). Use a number close to this mean for the number of neurons in the first hidden layer. Repeat this pattern for subsequent hidden layers ((number of neurons in the prior hidden layer + number of neurons in output layer) / 2). Softmax is the activation for the output layer that is used for multi-class classification. Categorial cross entropy and predictive model accuracy are respectively the loss functions and metrics used for multi-class classification*

# In[15]:


# Initialize the Deep Learning Neural Network model

nn_v0 = Sequential()


# In[16]:


# Design the network architecture 

# Define the model - deep neural net
number_input_features = len(X.columns)
number_output = 4

# Define hidden layers
i = 0
hidden_nodes_layer=(number_input_features+number_output)/2
while hidden_nodes_layer/2 > 4:
    if i == 0:
        nn_v0.add(Dense(units=round(hidden_nodes_layer), input_dim=number_input_features, activation='relu'))
        i+=1
    else:
        hidden_nodes_layer = (hidden_nodes_layer+number_output)/2
        nn_v0.add(Dense(units=round(hidden_nodes_layer), activation='relu'))
        i+=1

# Define output layer
nn_v0.add(Dense(units=number_output, activation='softmax'))

# Compile the model
nn_v0.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# display(nn_v0.summary())


# In[17]:


# Fit the data to the model

model_v0 = nn_v0.fit(X_train, y_train, epochs=100)


# In[18]:


# Plot the loss over epochs

plt.plot(model_v0.history["loss"])
plt.title("Model V0 Training Loss Function")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[19]:


# Plot the accuracy over epochs

plt.plot(model_v0.history["accuracy"])
plt.title("Model V0 Training Accuracy")
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.show()


# In[20]:


# Evaluate model on test set

model_loss, model_accuracy = nn_v0.evaluate(
    X_test, y_test, verbose=2
)
# print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# In[21]:


# # Have the neural network cast its prediction on what pitch is next

y_pred = nn_v0.predict(X_test)
pred_final = np.argmax(y_pred, axis=1)

# Translate target of test set into pitch type

y_test_reverted = []
for lists in y_test:
    if lists[0] == 1:
        y_test_reverted.append('CH')
    elif lists[1] == 1:
        y_test_reverted.append('CU')
    elif lists[2] == 1:
        y_test_reverted.append('FF')
    else:
        y_test_reverted.append('SL')

# Translate results into pitch type

y_pred_converted = []
for numbers in pred_final:
    if numbers == 0:
        y_pred_converted.append('CH')
    elif numbers == 1:
        y_pred_converted.append('CU')
    elif numbers == 2:
        y_pred_converted.append('FF')
    else: 
        y_pred_converted.append('SL')

# Place results in dataframe

final_results = pd.DataFrame({
    'Predictions': y_pred_converted,
    'Actual':  y_test_reverted})

# display(final_results.head())
# print(classification_report(final_results['Actual'], final_results['Predictions']))


# ## Neural Network Feature Importance Instance
# 
# Using the Lime library, visualize the importance of features for an instance in the testing data

# In[22]:


# Use Lime library to help visualize important features 

from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data = np.array(X_train), 
    feature_names = list(X_train.columns),
    class_names=['CH', 'CU', 'FF', 'SL'],
    mode='classification',
    verbose = True, 
    random_state=1
)

lime_exp = lime_explainer.explain_instance(
    data_row = X_test.iloc[0, :],
    predict_fn = nn_v0.predict,
    num_features=10 
)

# lime_exp.as_pyplot_figure()
# display(pd.DataFrame(lime_exp.as_list()))


# ## Optimize the Model
# 
# ### Optimized Model 1
# Optimize the model by increasing the number of epochs 

# In[23]:


#Define the new, optimized model

# nn_v1 = Sequential()


# # In[24]:


# # Design the network architecture 

# # Define the model - deep neural net
# number_input_features = len(X.columns)
# number_output = 4

# # Define hidden layers
# i = 0
# hidden_nodes_layer=(number_input_features+number_output)/2
# while hidden_nodes_layer/2 > 4: 
#     if i == 0:
#         nn_v1.add(Dense(units=round(hidden_nodes_layer), input_dim=number_input_features, activation='relu'))
#         i+=1
#     else:
#         hidden_nodes_layer = (hidden_nodes_layer+number_output)/2
#         nn_v1.add(Dense(units=round(hidden_nodes_layer), activation='relu'))
#         i+=1

# # Define output layer
# nn_v1.add(Dense(units=number_output, activation='softmax'))

# # Compile the model
# nn_v1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # print(len(X.columns))
# # display(nn_v1.summary())


# # In[25]:


# # Fit the data to the model

# model_v1 = nn_v1.fit(X_train, y_train, epochs=500)


# # In[26]:


# # Plot the loss over epochs

# plt.plot(model_v1.history["loss"])
# plt.title("Model V1 Training Loss Function")
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()


# # In[27]:


# # Plot the accuracy over epochs

# plt.plot(model_v1.history["accuracy"])
# plt.title("Model V1 Training Accuracy Function")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()


# # In[28]:


# # Evaluate model on test set

# model_loss, model_accuracy = nn_v1.evaluate(
#     X_test, y_test, verbose=2
# )
# # print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# # In[29]:


# # Have the neural network cast its prediction on what pitch is next

# y_pred = nn_v1.predict(X_test)
# pred_final = np.argmax(y_pred, axis=1)

# # Translate results into pitch type

# y_pred_converted = []
# for numbers in pred_final:
#     if numbers == 0:
#         y_pred_converted.append('CH')
#     elif numbers == 1:
#         y_pred_converted.append('CU')
#     elif numbers == 2:
#         y_pred_converted.append('FF')
#     else: 
#         y_pred_converted.append('SL')

# # Place results into dataframe

# final_results_v1 = pd.DataFrame({
#     'Predictions': y_pred_converted,
#     'Actual':  y_test_reverted})

# # display(final_results_v1.head())
# # print(classification_report(final_results_v1['Actual'], final_results_v1['Predictions']))


# # ### Optimized Model 2
# # 
# # From Keras, use the stochastic gradient descent (SGD) optimizer that is an iterative method for optimizing an objective function with suitable smoothness properties

# # In[30]:


# # Define the model

# nn_v2 = Sequential()


# # In[31]:


# # Design the network architecture 

# import tensorflow as tf

# tf.keras.optimizers.Adadelta()
# #from keras.optimizers import Adadelta
# # Define the model - deep neural net

# total_neurons = len(X.columns)*(2/3)
# number_input_features = len(X.columns)
# number_output = 4

# # Define hidden layers
# i = 0
# hidden_nodes_layer=(number_input_features+number_output)/2
# while hidden_nodes_layer/2 > 4: 
#     if i == 0:
#         nn_v2.add(Dense(units=round(hidden_nodes_layer), input_dim=number_input_features, activation='relu'))
#         i+=1
#     else:
#         hidden_nodes_layer = (hidden_nodes_layer+number_output)/2
#         nn_v2.add(Dense(units=round(hidden_nodes_layer), activation='relu'))
#         i+=1

# # Define output layer
# nn_v2.add(Dense(units=number_output, activation='softmax'))

# # Compile the model
# nn_v2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# # display(nn_v2.summary())


# # In[32]:


# # Fit the data to the model

# model_v2 = nn_v2.fit(X_train, y_train, epochs=100)


# # In[33]:


# # Plot the loss over epochs

# plt.plot(model_v2.history["loss"])
# plt.title("Model V2 Training Loss Function")
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()


# # In[34]:


# # Plot the accuracy over epochs

# plt.plot(model_v2.history["accuracy"])
# plt.title("Model V2 Training Accuracy Function")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()


# # In[35]:


# # Evaluate model on test set

# model_loss, model_accuracy = nn_v2.evaluate(
#     X_test, y_test, verbose=2
# )
# # print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# # In[36]:


# # Have the neural network cast its prediction on what pitch is next

# y_pred = nn_v2.predict(X_test)
# pred_final = np.argmax(y_pred, axis=1)

# # Translate results into pitch type

# y_pred_converted = []
# for numbers in pred_final:
#     if numbers == 0:
#         y_pred_converted.append('CH')
#     elif numbers == 1:
#         y_pred_converted.append('CU')
#     elif numbers == 2:
#         y_pred_converted.append('FF')
#     else: 
#         y_pred_converted.append('SL')

# # Put results into dataframe

# final_results_v2 = pd.DataFrame({
#     'Predictions': y_pred_converted,
#     'Actual':  y_test_reverted})

# # display(final_results_v2.head())
# # print(classification_report(final_results_v2['Actual'], final_results_v2['Predictions']))


# # ### Optimized Model 3
# # 
# # From Keras, use the stochastic gradient descent (SGD) optimizer that is an iterative method for optimizing an objective function with suitable smoothness properties

# # In[37]:


# # Define the model

# nn_v3 = Sequential()


# # In[38]:


# # Design the network architecture 

# from keras.optimizers import SGD
# # Define the model - deep neural net

# number_input_features = len(X.columns)
# number_output = 4

# # Define hidden layers
# i = 0
# hidden_nodes_layer=(number_input_features+number_output)/2
# while hidden_nodes_layer/2 > 4: 
#     if i == 0:
#         nn_v3.add(Dense(units=round(hidden_nodes_layer), input_dim=number_input_features, activation='relu'))
#         i+=1
#     else:
#         hidden_nodes_layer = (hidden_nodes_layer+number_output)/2
#         nn_v3.add(Dense(units=round(hidden_nodes_layer), activation='relu'))
#         i+=1

# # Define output layer
# nn_v3.add(Dense(units=number_output, activation='softmax'))

# # Compile the model
# nn_v3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# # display(nn_v3.summary())


# # In[39]:


# # Fit the data to the model

# model_v3 = nn_v3.fit(X_train, y_train, epochs=100)


# # In[40]:


# # Plot the loss over epochs

# plt.plot(model_v3.history["loss"])
# plt.title("Model V3 Training Loss Function")
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()


# # In[41]:


# # Plot the accuracy over epochs

# plt.plot(model_v3.history["accuracy"])
# plt.title("Model V3 Training Accuracy Function")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()


# # In[42]:


# # Evaluate model on test set

# model_loss, model_accuracy = nn_v3.evaluate(
#     X_test, y_test, verbose=2
# )
# # print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# # In[43]:


# # Have the neural network cast its prediction on what pitch is next

# y_pred = nn_v3.predict(X_test)
# pred_final = np.argmax(y_pred, axis=1)

# # Translate results into pitch type

# y_pred_converted = []
# for numbers in pred_final:
#     if numbers == 0:
#         y_pred_converted.append('CH')
#     elif numbers == 1:
#         y_pred_converted.append('CU')
#     elif numbers == 2:
#         y_pred_converted.append('FF')
#     else: 
#         y_pred_converted.append('SL')

# # Create into dataframe
# final_results_v3 = pd.DataFrame({
#     'Predictions': y_pred_converted,
#     'Actual':  y_test_reverted})

# display(final_results_v3.head())
# print(classification_report(final_results_v3['Actual'], final_results_v3['Predictions']))


# ## Field Testing
# 
# Justin Verlander pitched against the Arizona Diamondbacks on the evening of September 28, 2022 at home. The data from that game was extracted from Baseball Savant and used to test the 4 models above

# In[45]:


field_df = pd.read_csv(Path("resources/field_test_data.csv"))

field_df = field_df.loc[:,        ['pitch_type',
                                 'pitch_name',
                                   'game_date',
                                   'description',
                                #    'zone',
                                   'stand',
                                   'p_throws',
                                   'type',
                                   'balls',
                                   'strikes',
                                   'on_3b',
                                   'on_2b',
                                   'on_1b',
                                   'outs_when_up',
                                   'pitch_number',
                                   'bat_score',
                                   'fld_score',
                                   'if_fielding_alignment',
                                   'of_fielding_alignment'
                                  ]]

# display(verlander_df.columns)


# ## Feature Engineering

# In[46]:


# Shift pitches so that model does not know the upcoming pitch 

field_df['pitch_name'] = field_df['pitch_name'].shift(-1).dropna()
field_df['type'] = field_df['type'].shift(-1).dropna()

# display(field_df)


# In[47]:


# Converting batting score and fielding score to one column.
# Positive number means fielding team is winning and negative number means batting team is winning. 
field_df['score_diff'] = field_df['fld_score'] - field_df['bat_score']


# Drop batting score and fielding score columns now that you have the score differential
field_df.drop(columns = ['bat_score', 'fld_score'], inplace=True)


# In[48]:


# Feature engineering to count the number of pitches JV has thrown each outing

field_df['ones'] = 1
pitch_count_df = field_df[['game_date', 'ones']]
pitch_count_df['pitch_count'] = pitch_count_df.groupby(['game_date']).cumcount(ascending = False)
pitch_count_df['pitch_count'] = pitch_count_df['pitch_count'] + 1

field_df = pd.concat([field_df, pitch_count_df['pitch_count']], join='inner', axis=1)


# In[49]:


# Feature engineering to change the ball and strike count into one column as a string

field_df['count'] = field_df['balls'].astype(str) +'-'+ field_df['strikes'].astype(str)
field_df.drop(columns=['balls', 'strikes'], inplace=True)
field_df.head()


# In[50]:


# Encode 1st, 2nd and 3rd bases with 1s and 0s

# Fill NaN to 0 
field_df['on_3b'] = field_df['on_3b'].fillna(0)
field_df['on_2b'] = field_df['on_2b'].fillna(0)
field_df['on_1b'] = field_df['on_1b'].fillna(0)

# Change batter IDs to 1 
field_df['on_3b'][field_df['on_3b'] > 0.0] = 1.0
field_df['on_2b'][field_df['on_2b'] > 0.0] = 1.0
field_df['on_1b'][field_df['on_1b'] > 0.0] = 1.0

# display(field_df.head())


# 

# In[51]:


# Split into X and y 

X_real_testing = field_df.drop(columns=['pitch_type', 'game_date'])
y_real_testing = field_df['pitch_type']


# In[52]:


# Use get_dummies to encode categorical variables 

X_real_testing = pd.get_dummies(X_real_testing)

encoder = LabelEncoder()
encoder.fit(y_real_testing)
encoded_y_real = encoder.transform(y_real_testing)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_real = np_utils.to_categorical(encoded_y_real)

# if a parameter is not present in the game data, put in a column of 0s
parameters_list = {
    'stand_L',
    'stand_R',
    'if_fielding_alignment_Infield shift',
    'if_fielding_alignment_Standard',
    'if_fielding_alignment_Strategic',
    'of_fielding_alignment_Standard',
    'of_fielding_alignment_Strategic',
    'description_foul_bunt',
    'description_blocked_ball'
}
for string in parameters_list:
    if string not in X_real_testing:
        X_real_testing[string] = 0

# display(X_real_testing.columns)


# 

# In[53]:


# Evaluate model 1 on test set

model_loss, model_accuracy = nn_v0.evaluate(
    X_real_testing, dummy_y_real, verbose=2
)
# print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# In[54]:


# Evaluate model 2 on test set

# model_loss, model_accuracy = nn_v1.evaluate(
#     X_real_testing, dummy_y_real, verbose=2
# )
# # print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# # In[55]:


# # Evaluate model 3 on test set

# model_loss, model_accuracy = nn_v2.evaluate(
#     X_real_testing, dummy_y_real, verbose=2
# )
# # print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# # In[56]:


# # Evaluate model 4 on test set

# model_loss, model_accuracy = nn_v3.evaluate(
#     X_real_testing, dummy_y_real, verbose=2
# )
# print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# In[58]:


# # Have the neural network cast its prediction on what pitch is next

y_pred_v0 = nn_v0.predict(X_real_testing)
# y_pred_v1 = nn_v1.predict(X_real_testing)
# y_pred_v2 = nn_v2.predict(X_real_testing)
# y_pred_v3 = nn_v3.predict(X_real_testing)

pred_final_v0 = np.argmax(y_pred_v0, axis=1)
# pred_final_v1 = np.argmax(y_pred_v1, axis=1)
# pred_final_v2 = np.argmax(y_pred_v2, axis=1)
# pred_final_v3 = np.argmax(y_pred_v3, axis=1)

# Translate target of test set into pitch type

y_test_reverted = []
for lists in y_test:
    if lists[0] == 1:
        y_test_reverted.append('CH')
    elif lists[1] == 1:
        y_test_reverted.append('CU')
    elif lists[2] == 1:
        y_test_reverted.append('FF')
    else:
        y_test_reverted.append('SL')

# Place results in dataframe

final_results_real = pd.DataFrame()

# Translate results into pitch type

# for arrays in [pred_final_v0, pred_final_v1, pred_final_v2, pred_final_v3]:
#     y_pred_converted = []
#     for numbers in arrays:
#         if numbers == 0:
#             y_pred_converted.append('CH')
#         elif numbers == 1:
#             y_pred_converted.append('CU')
#         elif numbers == 2:
#             y_pred_converted.append('FF')
#         else: 
#             y_pred_converted.append('SL')
#     y_pred_series = pd.DataFrame(y_pred_converted)
#     final_results_real = pd.concat([final_results_real, y_pred_series], axis=1)

# final_results_real = pd.concat([final_results_real, y_real_testing], axis=1)
# final_results_real.columns = ['Model 1 (v0) Pred', 'Model 2 (v1) Pred', 'Model 3 (v2) Pred', 'Model 4 (v3) Pred', 'Actual']

# display(final_results_real)
# print(classification_report(final_results_real['Actual'], final_results_real['Model 1 (v0) Pred']))
# print(classification_report(final_results_real['Actual'], final_results_real['Model 2 (v1) Pred']))
# print(classification_report(final_results_real['Actual'], final_results_real['Model 3 (v2) Pred']))
# print(classification_report(final_results_real['Actual'], final_results_real['Model 4 (v3) Pred']))


# In[61]:


odds = pd.DataFrame(y_pred_v0, columns=["Changeup", "Curveball", "Fastball", "Slider"])

# display(odds)

