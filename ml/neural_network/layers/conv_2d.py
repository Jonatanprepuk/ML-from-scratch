import numpy as np
from ml.base import Trainable
from .base import Layer

class Conv2D(Layer, Trainable):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: tuple[int, int]=(1,1),
                 weight_regularizer_l1: float = 0, weight_regularizer_l2: float = 0,
                 bias_regularizer_l1: float = 0, bias_regularizer_l2: float = 0):
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else: 
            kh, kw = kernel_size
        
        limit = np.sqrt(2 / (in_channels * kh * kw))
        self.weights = np.random.randn(out_channels, in_channels, kh, kw) * limit
        self.biases = np.zeros((out_channels, 1))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride_w, self.stride_h = stride
        
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
        self.name =f'Conv2d_{id(self)}'
        
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.inputs = inputs
        cols, OH, OW, = self._im2col(inputs, self.kernel_size)
        w_col = self.weights.reshape(self.out_channels, -1).T
        out = cols @ w_col + self.biases.T
        
        self.output = out.reshape(inputs.shape[0], OH, OW, self.out_channels).transpose(0, 3, 1, 2)
        self.cols = cols
        
    def backward(self, dvalues: np.ndarray) -> None:
        batch_size, _, output_height, output_width = dvalues.shape

        dvalues_reshaped = dvalues.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.dbiases = np.sum(dvalues_reshaped, axis=0, keepdims=True).T

        dweights_col = self.cols.T @ dvalues_reshaped
        self.dweights = dweights_col.T.reshape(self.weights.shape)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        w_col = self.weights.reshape(self.out_channels, -1)
        dcols = dvalues_reshaped @ w_col
        self.dinputs = self._col2im(dcols, self.inputs.shape, self.kernel_size)
    
    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {f'{self.name}_weights': self.weights, f'{self.name}_biases': self.biases}

    @property
    def gradients(self) -> dict[str, np.ndarray]:
        return {f'{self.name}_weights' : self.dweights, f'{self.name}_biases' : self.dbiases}
    
    def set_parameter(self, name, value):
        if '_' in name:
            name = name.split('_', 2)[2]
            
        setattr(self, name, value)
    
    def _im2col(self, inputs: np.ndarray, kernel_size: int):
        batch_size, channels, input_height, input_width = inputs.shape
        kernel_height, kernel_width = kernel_size

        output_height = (input_height - kernel_height) // self.stride_h + 1
        output_width = (input_width - kernel_width) // self.stride_w + 1

        patch_list = []

        for out_y in range(output_height):
            for out_x in range(output_width):
                patch = inputs[:, :, 
                           out_y * self.stride_h : out_y * self.stride_h + kernel_height,
                           out_x * self.stride_w : out_x * self.stride_w + kernel_width]

                patch_flat = patch.reshape(batch_size, -1)
                patch_list.append(patch_flat)

        all_patches = np.stack(patch_list, axis=1)

        col_matrix = all_patches.reshape(-1, all_patches.shape[-1])

        return col_matrix, output_height, output_width

    def _col2im(self, cols: np.ndarray, input_shape: tuple, kernel_size: tuple) -> np.ndarray:
        batch_size, channels, input_height, input_width = input_shape
        kernel_height, kernel_width = kernel_size

        output_height = (input_height - kernel_height) // self.stride_h + 1
        output_width = (input_width - kernel_width) // self.stride_w + 1

        dinputs = np.zeros(input_shape)
        cols_reshaped = cols.reshape(batch_size, output_height * output_width, -1)
        cols_split = np.split(cols_reshaped, output_height * output_width, axis=1)

        idx = 0
        for out_y in range(output_height):
            for out_x in range(output_width):
                patch = cols_split[idx].reshape(batch_size, channels, kernel_height, kernel_width)
                dinputs[:, :, 
                        out_y * self.stride_h : out_y * self.stride_h + kernel_height,
                        out_x * self.stride_w : out_x * self.stride_w + kernel_width] += patch
                idx += 1

        return dinputs