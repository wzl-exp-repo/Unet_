
#from Segmentation.preprocess import Data_Preprocess



#test = Data_Preprocess()
#test.get_dataset()
import numpy as np


class Model(object):

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self._prediction = None
        self._optimize = None
        self._error = None

    @property
    def prediction(self):
        #if  not self._prediction:
        print('test')
            
        #self._prediction = 24
        return self._prediction

    @prediction.setter
    def prediction(self,value):
        if  not self._prediction:
            print('Test')
            
            self._prediction = value
        #return self._prediction





class C(object):
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        if self._x == None:
           self._x = 1000
        return self._x

    @x.setter
    def x(self, value):
        self._x = value


m = Model(np.array([[1,2],[3,4]]),np.array([1,2]))
v = C()

#print(m._prediction)
print(v.x)
