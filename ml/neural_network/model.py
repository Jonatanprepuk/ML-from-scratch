import numpy as np
import pickle
import copy

from .layers import  Layer, LayerInput
from .activations import Softmax
from .accuracies import Accuracy
from ..losses import Loss, CategoricalCrossentropy, SoftmaxCategoricalCrossentropy
from ..optimizers import Optimizer

class Model:
    def __init__(self, *layers: Layer) -> None:
        self.layers = []
        self.softmax_classifier_output = None 
        
        self.val_accuracy_history = []
        self.val_loss_history = []
        
        for layer in layers:
            self.add(layer)
        
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
    
    def set(self, *, loss: Loss=None, optimizer: Optimizer=None, accuracy: Accuracy= None) -> None:
        self.loss = loss
        self.optimizer  = optimizer
        self.accuracy = accuracy
    
    def finalize(self) -> None:
        self.input_layer = LayerInput()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossentropy):
            self.softmax_classifier_output = SoftmaxCategoricalCrossentropy()
    
    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int=1, batch_size: int=None,
              print_every: int=1, validation_data: np.ndarray=None) -> None:
        self.loss_history = []
        self.accuracy_history = []
        
        self.accuracy.init(y)

        train_steps = 1 

        if validation_data is not None:
            validation_steps = 1

            X_val, y_val = validation_data
        
        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1 

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs+1):

            print(f'epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step + 1) * batch_size]
                    batch_y = y[step*batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training=True)
            
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
        
                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                if not step % print_every or step == train_steps -1:
                    print(f'step: {step}, ' + 
                          f'acc: {accuracy:.3f} '+
                          f'loss: {loss:.3f} ' +
                          f'data_loss: {data_loss:.3f} '+
                          f'reg_loss: {regularization_loss:.3f} ' +
                          f'lr: {self.optimizer.current_learning_rate}')
                    

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

                
            print(f'training, ' + 
                  f'acc: {epoch_accuracy:.3f} '+
                  f'loss: {epoch_loss:.3f} ' +
                  f'data_loss: {epoch_data_loss:.3f} '+
                  f'reg_loss: {epoch_regularization_loss:.3f} ' +
                  f'lr: {self.optimizer.current_learning_rate}')
            
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)
                
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)
                    
    def predict(self, X: np.ndarray, *, batch_size: int=None) -> np.ndarray:
        prediction_steps = 1
        
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        output = []
        
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
            
        return np.vstack(output)

    def forward(self, X: np.ndarray, training: bool) -> np.ndarray:
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output
    
    def backward(self, output: np.ndarray, y: np.ndarray) -> None:

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray, *, batch_size: int=None) -> None:
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):

            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step + 1) * batch_size]
                batch_y = y_val[step*batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
            
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
            
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f} ' +
              f'loss: {validation_loss:.3f}')
        
        self.val_accuracy_history.append(validation_accuracy)
        self.val_loss_history.append(validation_loss)
    
    def summary(self, input_shape: int= None) -> None:
        if not hasattr(self, 'input_shape'):
            if input_shape is None:
                raise ValueError("Input shape is unknown. Provide 'input_shape' or run the model once.")
                
            self.input_shape = input_shape
            X_dummy = np.zeros((1, self.input_shape))
            self.forward(X_dummy, training=False)
        else: 
            input_shape = self.input_shape
            
        X_dummy = np.zeros((1, self.input_shape))
        self.forward(X_dummy, training=False)

        layer_names = [type(layer).__name__ for layer in self.layers]
        output_shapes = [f"(None, {layer.output.shape[1]})" if hasattr(layer, 'output') else "-" for layer in self.layers]
        param_counts = [np.prod(layer.weights.shape) + np.prod(layer.biases.shape)if hasattr(layer, 'weights') else 0 for layer in self.layers]

        name_width = max(len(name) for name in layer_names)
        shape_width = max(len(shape) for shape in output_shapes)
        shape_width = max(shape_width, 12)
        param_width = max(len(str(p)) for p in param_counts)
        param_width = max(param_width, 6)

        print("\u2554" + "═" * (6 + name_width + shape_width + param_width + 7) + "╗")
        print("\u2551{:^{width}}║".format("Model Architecture", width=(6 + name_width + shape_width + param_width + 7)))
        print("\u2560" + "═" * 4 + "╬" + "═" * (name_width + 2) + "╬" + "═" * (shape_width + 2) + "╬" + "═" * (param_width + 2) + "╣")
        print(f"║ #  ║ {'Layer':<{name_width}} ║ {'Output Shape':<{shape_width}} ║ {'Params':>{param_width}} ║")
        print("\u2560" + "═" * 4 + "╬" + "═" * (name_width + 2) + "╬" + "═" * (shape_width + 2) + "╬" + "═" * (param_width + 2) + "╣")

        total_params = 0
        for i, (name, shape, params) in enumerate(zip(layer_names, output_shapes, param_counts)):
            print(f"║ {i+1:<2} ║ {name:<{name_width}} ║ {shape:<{shape_width}} ║ {params:>{param_width}} ║")
            total_params += params

        print("\u255A" + "═" * 4 + "╩" + "═" * (name_width + 2) + "╩" + "═" * (shape_width + 2) + "╩" + "═" * (param_width + 2) + "╝")
        print(f"Total trainable parameters: {total_params:,}")
        
    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_history, label="Training Loss")
        
        if self.val_loss_history is not None:
            plt.plot(self.val_loss_history, label="Validation Loss")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_parameters(self) -> list:
        parameters = []
        
        for layer in self.trainable_layers:
            layer_parameters = layer.parameters
            weights = layer_parameters[f'{layer.name}_weights']
            biases = layer_parameters[f'{layer.name}_biases']
            parameters.append((weights, biases))
        
        return parameters
    
    def set_parameters(self, parameters: list) -> None:
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            weights, biases = parameter_set
    
            layer.set_parameter("weights", weights)
            layer.set_parameter("biases", biases)
            
    def save_parameters(self, path:str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    def load_parameters(self, path: str):
        with open(path, 'rb') as f: 
            self.set_parameters(pickle.load(f))
        
    def save(self, path: str) -> None:
        model = copy.deepcopy(self)
        
        model.loss.new_pass()
        model.accuracy.new_pass()
        
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        with open(path, 'wb') as f: 
            pickle.dump(model, f)

    @staticmethod
    def load(path: str) -> 'Model':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
            