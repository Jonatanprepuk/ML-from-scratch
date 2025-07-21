import numpy as np
from .base import Layer

class MaxPooling2d(Layer):
    #TODO Optimera - nuvarande lösning är extremt långsam pga näslade for-loopar. 
    def __init__(self, *, pool_size:tuple[int, int]=(2, 2), stride:tuple[int, int]=(1, 1)):
        self.pool_h, self.pool_w = pool_size
        self.stride_h, self.stride_w = stride
         
    
    def forward(self, inputs:np.ndarray, training:bool):
        self.inputs = inputs
        batch_size, channels, height, width = inputs.shape
        
        output_height = (height - self.pool_h) // self.stride_h + 1
        output_width = (width - self.pool_w) // self.stride_w + 1
        
        output = np.zeros((batch_size, channels, output_height, output_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride_h
                        h_end = h_start + self.pool_h
                        w_start = j * self.stride_w
                        w_end = w_start + self.pool_w
                        
                        patch = inputs[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(patch)
        
        self.output = output
                        
    
    def backward(self, dvalues):
        batch_size, channels, output_height, output_width = dvalues.shape
        dinputs = np.zeros(self.inputs.shape)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride_h
                        h_end = h_start + self.pool_h
                        w_start = j * self.stride_w
                        w_end = w_start + self.pool_w
                        patch = self.inputs[b, c, h_start:h_end, w_start:w_end]
                        
                        max_idx = np.argmax(patch)
                        max_pos = np.unravel_index(max_idx, patch.shape)
                        
                        dinputs[b, c, h_start + max_pos[0], w_start + max_pos[1]] += dvalues[b, c, i, j]
        
        self.dinputs = dinputs
                        
        