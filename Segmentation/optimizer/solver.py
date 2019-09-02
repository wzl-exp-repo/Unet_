#Doniyor Tropmann


#It's old version of solver. it is not used.




#from __future__ import print_function, division
#from future import standard_library
#standard_library.install_aliases()

from builtins import object
import os

import json
import logging

import tensorflow as tf
from keras import losses

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers





class Solver(object):

    def __init__(self, model, data_train,data_val, **kwargs):


        self.model = model
        self.data_train = data_train
        self.data_val   = data_val
        self.num_all_train_examples = kwargs.pop('num_train_examples')
        self.num_all_val_examples = kwargs.pop('num_val_examples')

        # Unpack keyword arguments
        #self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.batch_size = kwargs.pop('batch_size', 10)
        self.num_epochs = kwargs.pop('num_epochs', 2)

        self._loss = kwargs.pop('loss', 'bce_dice_loss')
        self._optimizer = kwargs.pop('optimizer', 'adam')
        self._optim_config = kwargs.pop('optimizer_config', {})

        '''
        self.epsilon    = kwargs.pop('epsilon', 0.000000001)
        self.learning_rate = kwargs.pop('learning_rate', 1e-5)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.adam_eps   =kwargs.pop('adam_eps',  0.00001)
        self.threads    = kwargs.pop('threads', 4)
        self.learning_rate_step = kwargs.pop('learning_rate_step', null)
        self.max_steps = kwargs.pop('max_steps', 12000)
        '''


        self._metrics    = kwargs.pop('metrics', ['dice_loss'])
        self.metrics_list = {
                            'dice_loss': self.dice_loss
                            ,'f1': self.F1
                            ,'recall': self.recall
                            ,'precision': self.precision
                           }

        self._save_model_path = kwargs.pop('save_model_path', None)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)


    @property
    def loss(self):
        if self._loss == 'bce_dice_loss':
            print('Here Loss')

            self._loss = self.bce_dice_loss

        elif self._loss not in [self.bce_dice_loss]:
           print(type(self._loss))
           logging.error('Unrecognized loss type')
        return self._loss


    @property
    def optimizer(self):
        if self._optimizer == 'rms':
            if not self._optim_config:
                self._optim_config = {
                    'lr': 1e-5,
                    'decay': 0.9,
                    'rho': 0.9,
                    'epsilon': 1e-10
                }
            self._optimizer = optimizers.RMSprop(**self._optim_config)
            #self._optimizer = tf.train.RMSPropOptimizer(self._optim_config)
        elif self._optimizer == 'adam':
            if not self._optim_config:
                self._optim_config = {
                    'lr': 1e-5,
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'epsilon': 1e-08,
                    'decay' :  0.0,
                    'amsgrad' : False
                }
            self._optimizer = optimizers.Adam(**self._optim_config)
            #self._optimizer = tf.train.AdamOptimizer(self._optim_config)
        elif self._optimizer == 'sgd':
            if not self._optim_config:
                self._optim_config = {
                    'lr': 1e-5,
                    'momentum': 0.0,
                    'decay': 0.8,
                    'nesterov': False
                }
            self._optimizer = optimizers.SGD(**self._optim_config)
            #self._optimizer = tf.train.GradientDescentOptimizer(self._optim_config)

        elif type(self._optimizer) not in [optimizers.Adam, optimizers.SGD, optimizers.RMSprop]:
           logging.error('Unrecognized optimizer type')

        return self._optimizer


    @property
    def metrics(self):
        if not all(metric in self.metrics_list.values() for metric in self._metrics):
            self._metrics = [self.metrics_list[m] for m in self._metrics if m in self.metrics_list]

        if not self._metrics:
            logging.error('Unrecognized metric type')
        return self._metrics

    @property
    def model_dir(self):
        if not self._save_model_path:
            self._save_model_path = os.path.join(os.getcwd(),'weights.hdf5')
        return self._save_model_path



    def train(self):
        """
        Run optimization to train the model.
        """
        print('Here+++++++++++++++++++++++++++++=')
        num_train_examples = self.num_all_train_examples
        num_val_examples   = self.num_all_val_examples
        iterations_per_epoch = max(num_train_examples // self.batch_size, 1)
        validation_steps     = max(num_val_examples // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        cp = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_dir, monitor='val_dice_loss', save_best_only=True,
                                                verbose=self.verbose)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)


        print('Begin')
        history = self.model.fit(self.data_train,
                                 steps_per_epoch=iterations_per_epoch,
                                 epochs=self.num_epochs,
                                 validation_data=self.data_val,
                                 validation_steps=validation_steps,
                                 callbacks=[cp]
                                )
        print('End')

        try:
           logging.info('Saving train history')
           self.save_train_history(history)
        except:
            logging.error('Saving of history fail')

        return history

    def save_train_history(self, history):
        try:
            with open('train_history.txt', 'w') as file:
                file.write(json.dumps(history.history))
        except:
            logging.error("Could not save train history.")


    def dice_loss(self, mask_true, mask_pred):
        def dice_coeff(mask_true, mask_pred):
            smooth = 1.
            # Flatten
            mask_true_f = tf.reshape(mask_true, [-1])
            mask_pred_f = tf.reshape(mask_pred, [-1])
            intersection = tf.reduce_sum(mask_true_f * mask_pred_f)
            score = (2. * intersection + smooth) / (tf.reduce_sum(mask_true_f) + tf.reduce_sum(mask_pred_f) + smooth)
            return score

        return 1 - dice_coeff(mask_true, mask_pred)

    def bce_dice_loss(self, mask_true, mask_pred):
        # Compute balanced_dice_loss

        balanced_dice_loss = losses.binary_crossentropy(mask_true, mask_pred) + self.dice_loss(mask_true, mask_pred)

        return balanced_dice_loss


    def recall(self,mask_true, mask_pred):
        true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(mask_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(self, mask_true, mask_pred):
        true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def F1(self, mask_true, mask_pred):
        precision_val = self.precision(mask_true, mask_pred)
        recall_val = self.recall(mask_true, mask_pred)
        return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))
