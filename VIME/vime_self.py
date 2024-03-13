"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

vime_self.py
- Self-supervised learning parts of the VIME framework
- Using unlabeled data to train the encoder
"""

# Necessary packages
from keras.layers import Input, Dense
from keras.models import Model
from keras import models
from tensorflow import keras

from vime_utils import mask_generator, pretext_generator


def vime_self (x_unlab, p_m, alpha, parameters):
  """Self-supervised learning part in VIME.
  
  Args:
    x_unlab: unlabeled feature
    p_m: corruption probability
    alpha: hyper-parameter to control the weights of feature and mask losses
    parameters: epochs, batch_size
    
  Returns:
    encoder: Representation learning block
  """
    
  # Parameters
  _, dim = x_unlab.shape
  epochs = parameters['epochs']
  batch_size = parameters['batch_size']
  
  # Build model  
  inputs = Input(shape=(dim,))
  # Encoder  
  #h = Dense(int(dim), activation='relu')(inputs)
  #h = Dense(512, activation='relu')(inputs)
  h1 = Dense(512, activation='relu')(inputs)
  h2 = Dense(512, activation='relu')(h1)
  h3 = Dense(512, activation='relu')(h2)
  h4 = Dense(512, activation='relu')(h3)
  # Mask estimator
  #output_1 = Dense(dim, activation='sigmoid', name = 'mask')(h) 
  output_1 = Dense(dim, activation='sigmoid', name = 'mask')(h4) 
  # Feature estimator
  #output_2 = Dense(dim, activation='sigmoid', name = 'feature')(h)
  output_2 = Dense(dim, activation='sigmoid', name = 'feature')(h4)
  
  model = Model(inputs = inputs, outputs = [output_1, output_2])
  
  model.compile(optimizer='rmsprop',
                loss={'mask': 'binary_crossentropy', 
                      'feature': 'mean_squared_error'},
                loss_weights={'mask':1, 'feature':alpha})
  
  # Generate corrupted samples
  m_unlab = mask_generator(p_m, x_unlab)
  m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
  
  # Fit model on unlabeled data
  history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, 
            epochs = epochs, batch_size= batch_size)
      
  # Extract encoder part
  #layer_name = model.layers[1].name
  layer_name = model.layers[4].name
  layer_output = model.get_layer(layer_name).output
  encoder = models.Model(inputs=model.input, outputs=layer_output)
  
  return encoder, history


def vime_self_modified(x_unlab, p_m, alpha, parameters) :
    # parameters 
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    # Build model
    inputs = Input(shape=(dim,))
    # Encoder
    h1 = Dense(1024, activation='relu')(inputs)
    h2 = Dense(512, activation='relu')(h1)
    h3 = Dense(256, activation='relu')(h2)
    h4 = Dense(128, activation='relu')(h3)
    h5 = Dense(64, activation='relu')(h4)
    # Mask Decoder
    m1 = Dense(64, activation='relu')(h5)
    m2 = Dense(128, activation='relu')(m1)
    m3 = Dense(256, activation='relu')(m2)
    m4 = Dense(512, activation='relu')(m3)
    m5 = Dense(1024, activation='relu')(m4)
    output_1 = Dense(dim, activation='sigmoid', name='mask')(m5)
    # Feature Decoder
    f1 = Dense(64, activation='relu')(h5)
    f2 = Dense(128, activation='relu')(f1)
    f3 = Dense(256, activation='relu')(f2)
    f4 = Dense(512, activation='relu')(f3)
    f5 = Dense(1024, activation='relu')(f4)
    output_2 = Dense(dim, activation='sigmoid', name='feature')(f5)
    
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    
    model = Model(inputs=inputs, outputs=[output_1, output_2])
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer='rmsprop', loss={'mask': 'binary_crossentropy','feature': 'mean_squared_error'},loss_weights={'mask':1, 'feature':alpha})
    
    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # Fit model on unlabeled data
    history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, epochs = epochs, batch_size=batch_size)
    
    # Extract Encoder
    layer_name = model.layers[5].name
    layer_output = model.get_layer(layer_name).output
    encoder = models.Model(inputs=model.input, outputs=layer_output)
    
    return encoder, history
  
  
def DAE(x_unlab, p_m, alpha, parameters) :
    # parameters 
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    # Build model
    inputs = Input(shape=(dim,))
    # Encoder
    h1 = Dense(1024, activation='relu')(inputs)
    h2 = Dense(512, activation='relu')(h1)
    h3 = Dense(256, activation='relu')(h2)
    h4 = Dense(128, activation='relu')(h3)
    h5 = Dense(64, activation='relu')(h4)

    # Feature Decoder
    f1 = Dense(64, activation='relu')(h5)
    f2 = Dense(128, activation='relu')(f1)
    f3 = Dense(256, activation='relu')(f2)
    f4 = Dense(512, activation='relu')(f3)
    f5 = Dense(1024, activation='relu')(f4)
    output_2 = Dense(dim, activation='sigmoid', name='feature')(f5)
    
    model = Model(inputs=inputs, outputs=output_2)
    model.compile(optimizer='rmsprop', loss={'feature': 'mean_squared_error'},loss_weights={'feature':1})
    
    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # Fit model on unlabeled data
    history = model.fit(x_tilde, {'feature': x_unlab}, epochs = epochs, batch_size=batch_size)
    
    # Extract Encoder
    layer_name = model.layers[5].name
    layer_output = model.get_layer(layer_name).output
    encoder = models.Model(inputs=model.input, outputs=layer_output)
    
    return encoder, history
