#Doniyor Tropmann

import os
from tensorflow.python.keras import models
import logging


from Segmentation.architecture.Unet.encoder import Encoder
from Segmentation.architecture.Unet.decoder import Decoder
from Segmentation.optimizer.solver_encoder_decoder import Solver

class assistant(object):

    def __init__(self, data_train, data_val, **kwargs):

        self.data_train = data_train
        self.data_val = data_val
        self._image_shape = kwargs.pop('image_shape', None)
        #self._inputs  = layers.Input(shape=self.image_shape)

        self.encoder = kwargs.pop('encoder',Encoder(self.image_shape))
        self.decoder = kwargs.pop('decoder',Decoder(**kwargs))
        self._solver  = None
        self.solver_params = kwargs
        self.solver_params['loss'] = self.decoder.loss
        #self.solver  = kwargs.pop('solver', Solver(self.build_model(),self.data_train, self.data_val, **kwargs))
        self._weights_path = kwargs.pop('save_model_path', None)
        '''
        self.num_all_train_examples = kwargs.pop('num_train_examples')
        self.num_all_val_examples = kwargs.pop('num_val_examples')

        self.batch_size = kwargs.pop('batch_size', 10)
        self.num_epochs = kwargs.pop('num_epochs', 2)
        self._weights_dir = kwargs.pop('weights_dir', None)

        self.loss = kwargs.pop('loss', 'bce_dice_loss')
        self.optimizer = kwargs.pop('optimizer', 'adam')
        self.optim_config = kwargs.pop('optimizer_config', {})
        self.metrics = kwargs.pop('metrics', ['dice_loss'])

        self.verbose = kwargs.pop('verbose', True)
        '''
    @property
    def solver(self):
        if not self._solver:
            model = self.build_model()
            self._solver = Solver(model, self.data_train, self.data_val, **self.solver_params)
        return self._solver

    @property
    def image_shape(self):
        if not self._image_shape:
            self._image_shape = (64,64,3)
        return self._image_shape

    @property
    def weights_path(self):
        if not self._weights_path:
            self._weights_path= os.path.join(os.getcwd(),'weights.hdf5')
        return self._weights_path

    def build_model(self):

            encoder = self.encoder.build_encoder()
            print('Encoder', encoder)
            model   = self.decoder.build_decoder(encoder)
            # model.compile(optimizer=optimizers, loss=bce_dice_loss, metrics=[dice_loss, precision, recall, F1])
            return model

    def run_training(self):
         solver = self.solver
         result = solver.train()

         return result


    def run_inference(self, weights_path, batch):
        print('Before')
        recovered_model = self.load_weights(weights_path)
        print('After')
        if recovered_model:
            print('In')
            return  recovered_model.predict(batch)
        print('Fail')




    def load_weights(self, weights_path):
           logging.info('Loading weights')
           model = models.load_model( weights_path
                                  ,custom_objects={
                                    'bce_dice_loss': self.decoder.bce_dice_loss
                                    , 'dice_loss': self.solver.dice_loss
                                    , 'precision': self.solver.precision
                                    , 'recall': self.solver.recall
                                    , 'F1': self.solver.F1
                                  }
                                 )
           return model
            #except:
            #logging.error("Can't load weights")



