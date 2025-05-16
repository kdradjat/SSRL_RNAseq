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
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras import models
from keras import backend as K
from keras import metrics
from tensorflow import keras
import tensorflow as tf
import math

from ssrl_rnaseq.vime.vime_utils import mask_generator, pretext_generator


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
    history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, epochs = epochs, batch_size=batch_size, validation_split=0.1)
    
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



####################### REWORK ##############################

def vime_self_baseline(x_unlab, p_m, alpha, parameters) :
    # parameters 
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    # Build model
    inputs = Input(shape=(dim,))
    # Encoder
    h1 = Dense(100, activation='relu')(inputs)
    b1 = BatchNormalization()(h1)
    h2 = Dense(100, activation='relu')(b1)
    b2 = BatchNormalization()(h2)
    # Mask Decoder
    m1 = Dense(100, activation='relu')(b2)
    m2 = Dense(100, activation='relu')(m1)
    output_1 = Dense(dim, activation='sigmoid', name='mask')(m2)
    # Feature Decoder
    f1 = Dense(64, activation='relu')(b2)
    f2 = Dense(128, activation='relu')(f1)
    output_2 = Dense(dim, activation='sigmoid', name='feature')(f2)
    
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    
    model = Model(inputs=inputs, outputs=[output_1, output_2])
    model.compile(optimizer='rmsprop', loss={'mask': 'binary_crossentropy','feature': 'mean_squared_error'},loss_weights={'mask':1, 'feature':alpha})
    
    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # Fit model on unlabeled data
    history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, epochs = epochs, batch_size=batch_size, validation_split=0.1)    
    
    # Extract Encoder
    layer_name = model.layers[4].name
    layer_output = model.get_layer(layer_name).output
    encoder = models.Model(inputs=model.input, outputs=layer_output)
    
    return encoder, history



# Data Generator to handle OOM issues
# Generator with no shuffle
class DataGenerator(Sequence):
    def __init__(self, x_set, m_set, x_true_set, batch_size):
        self.x = x_set
        self.m = m_set
        self.x_true = x_true_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = {'mask': self.m[idx * self.batch_size:(idx + 1) * self.batch_size], 'feature': self.x_true[idx * self.batch_size:(idx + 1) * self.batch_size]}
        return batch_x, batch_y
    
# Generator with shuffle by using indexes
class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, m_set, x_true_set, batch_size):
        self.x, self.m, self.x_true = x_set, m_set, x_true_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        #batch_y = self.y[inds]
        batch_y = {'mask': self.m[inds], 'feature': self.x_true[inds]}
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
def batch_generator(x_set, m_set, x_true_set, batch_size):
    idx = 0
    while True:
        # fill up the batch
        X_batch = x_set[idx * batch_size:(idx + 1) * batch_size]
        y1_batch = m_set[idx * batch_size:(idx + 1) * batch_size]
        y2_batch = x_true_set[idx * batch_size:(idx + 1) * batch_size]
        idx += 1
            
        yield(X_batch, {'mask': y1_batch, 'feature': y2_batch} )

def vime_self_4layers(x_unlab, p_m, alpha, parameters) :
    # parameters 
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    # Build model
    inputs = Input(shape=(dim,))
    # Encoder
    h1 = Dense(256)(inputs)
    b1 = BatchNormalization()(h1)
    b1 = Activation('relu')(b1)
    b1 = Dropout(0.2)(b1)
    h2 = Dense(256)(b1)
    b2 = BatchNormalization()(h2)
    b2 = Activation('relu')(b2)
    b2 = Dropout(0.2)(b2)
    h3 = Dense(256)(b2)
    b3 = BatchNormalization()(h3)
    b3 = Activation('relu')(b3)
    b3 = Dropout(0.2)(b3)
    h4 = Dense(256)(b3)
    b4 = BatchNormalization()(h4)
    b4 = Activation('relu')(b4)
    b4 = Dropout(0.2)(b4)
    # Mask Decoder
    m1 = Dense(256, activation='relu')(b4)
    m2 = Dense(256, activation='relu')(m1)
    m3 = Dense(256, activation='relu')(m2)
    m4 = Dense(256, activation='relu')(m3)
    output_1 = Dense(dim, activation='sigmoid', name='mask')(m4)
    # Feature Decoder
    f1 = Dense(256, activation='relu')(b4)
    f2 = Dense(256, activation='relu')(f1)
    f3 = Dense(256, activation='relu')(f2)
    f4 = Dense(256, activation='relu')(f3)
    output_2 = Dense(dim, activation='sigmoid', name='feature')(f4)
    
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    
    model = Model(inputs=inputs, outputs=[output_1, output_2])
    model.compile(optimizer='rmsprop', loss={'mask': 'binary_crossentropy','feature': 'mean_squared_error'},loss_weights={'mask':1, 'feature':alpha})
    
    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # data generator to handle OOM issues
    data_train = Generator(x_tilde, m_label, x_unlab, batch_size)
    
    # Fit model on unlabeled data
    #history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, epochs = epochs, batch_size=batch_size, validation_split=0.1)
    history = model.fit_generator(data_train, epochs = epochs)
    #history = model.fit_generator(batch_generator(x_tilde, m_label, x_unlab, batch_size), epochs=epochs)
    
    # Extract Encoder
    layer_name = model.layers[16].name
    layer_output = model.get_layer(layer_name).output
    encoder = models.Model(inputs=model.input, outputs=layer_output)
    
    del encoder
    del model
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    
    return encoder, history
    #return history


def vime_self_subtab(x_unlab, p_m, alpha, parameters) :
    # parameters 
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    # Build model
    inputs = Input(shape=(dim,))
    # Encoder
    h1 = Dense(1024, activation='relu')(inputs)
    b1 = BatchNormalization()(h1)
    h2 = Dense(784, activation='relu')(b1)
    b2 = BatchNormalization()(h2)
    h3 = Dense(784, activation='relu')(b2)
    b3 = BatchNormalization()(h3)
    # Mask Decoder
    m1 = Dense(784, activation='relu')(b3)
    m2 = Dense(784, activation='relu')(m1)
    m3 = Dense(1024, activation='relu')(m2)
    output_1 = Dense(dim, activation='sigmoid', name='mask')(m3)
    # Feature Decoder
    f1 = Dense(784, activation='relu')(b3)
    f2 = Dense(784, activation='relu')(f1)
    f3 = Dense(1024, activation='relu')(f2)
    output_2 = Dense(dim, activation='sigmoid', name='feature')(f3)
    
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    
    model = Model(inputs=inputs, outputs=[output_1, output_2])
    model.compile(optimizer='rmsprop', loss={'mask': 'binary_crossentropy','feature': 'mean_squared_error'},loss_weights={'mask':1, 'feature':alpha})
    
    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # Fit model on unlabeled data
    history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, epochs = epochs, batch_size=batch_size, validation_split=0.1)    
    
    # Extract Encoder
    layer_name = model.layers[6].name
    layer_output = model.get_layer(layer_name).output
    encoder = models.Model(inputs=model.input, outputs=layer_output)
    
    return encoder, history



def vime_self_custom(x_unlab, emb_size, depth, p_m, alpha, parameters) :
    # parameters 
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    inputs = Input(shape=(dim,))
    x = Dense(emb_size)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # Encoder
    for _ in range(depth-1) :
        x = Dense(emb_size)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        
    # Mask Decoder
    m = Dense(emb_size, activation='relu')(x)
    for _ in range(depth-1) :
        m = Dense(emb_size, activation='relu')(m)
    output_1 = Dense(dim, activation='sigmoid', name='mask')(m)
    
    # Feature Decoder
    f = Dense(emb_size, activation='relu')(x)
    for _ in range(depth-1):
        f = Dense(emb_size, activation='relu')(f)
    output_2 = Dense(dim, activation='sigmoid', name='feature')(f)
    
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    
    model = Model(inputs=inputs, outputs=[output_1, output_2])
    model.compile(optimizer='rmsprop', loss={'mask': 'binary_crossentropy','feature': 'mean_squared_error'},loss_weights={'mask':1, 'feature':alpha})
    
    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # Fit model on unlabeled data
    history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, epochs = epochs, batch_size=batch_size, validation_split=0.1)    
    
    # Extract Encoder
    layer_name = model.layers[depth*4].name
    layer_output = model.get_layer(layer_name).output
    encoder = models.Model(inputs=model.input, outputs=layer_output)
    
    return encoder, history


def dae(x_unlab, parameters) :
    # parameters 
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    # Build model
    inputs = Input(shape=(dim,))
    # Encoder
    h1 = Dense(256, activation='relu')(inputs)
    b1 = BatchNormalization()(h1)
    h2 = Dense(256, activation='relu')(b1)
    b2 = BatchNormalization()(h2)
    h3 = Dense(256, activation='relu')(b2)
    b3 = BatchNormalization()(h3)
    h4 = Dense(256, activation='relu')(b3)
    b4 = BatchNormalization()(h4)
    
    # Feature Decoder
    f1 = Dense(256, activation='relu')(b4)
    f2 = Dense(256, activation='relu')(f1)
    f3 = Dense(256, activation='relu')(f2)
    f4 = Dense(256, activation='relu')(f3)
    outputs = Dense(dim, activation='sigmoid', name='feature')(f4)
    
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='rmsprop', loss={'feature': 'mean_squared_error'})
    
    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # Fit model on unlabeled data
    history = model.fit(x_tilde, {'feature': x_unlab}, epochs = epochs, batch_size=batch_size, validation_split=0.1)    
    
    # Extract Encoder
    layer_name = model.layers[8].name
    layer_output = model.get_layer(layer_name).output
    encoder = models.Model(inputs=model.input, outputs=layer_output)
    
    return encoder, history