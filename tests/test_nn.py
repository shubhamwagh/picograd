import unittest

from picograd.engine import Var
from picograd.nn import Neuron, Layer, MLP


class TestNN(unittest.TestCase):
    def test_neuron(self):
        n_features = 3
        neuron = Neuron(in_features=n_features)

        # Check parameter count
        self.assertEqual(len(neuron.parameters()), n_features + 1)  # Including bias

        # Overwite parameters for output prediction
        neuron.w = list(Var(x) for x in range(n_features))
        neuron.b = Var(-0.75)

        # Check output computation (weighted sum + bias)

        # test without activation
        neuron.activation = None
        self.assertEqual(neuron([Var(1), Var(-0.5), Var(1.5)]).data, 1.75)  # 1*0 + (-0.5)*1 + 1.5*2 + 1*(-0.75)
        self.assertEqual(neuron([Var(1), Var(-3), Var(1.5)]).data, -0.75)  # 1*0 + (-3)*1 + 1.5*2 + 1*(-0.75)

        neuron.activation = 'linear'
        self.assertEqual(neuron([Var(1), Var(-0.5), Var(1.5)]).data, 1.75)  # 1*0 + (-0.5)*1 + 1.5*2 + 1*(-0.75)
        self.assertEqual(neuron([Var(1), Var(-3), Var(1.5)]).data, -0.75)  # 1*0 + (-3)*1 + 1.5*2 + 1*(-0.75)

        # test with relu activation function
        neuron.activation = 'relu'
        self.assertEqual(neuron([Var(1), Var(-0.5), Var(1.5)]).data, 1.75)
        self.assertEqual(neuron([Var(1), Var(-3), Var(1.5)]).data, 0)

        # test with tanh activation function
        neuron.activation = 'tanh'
        self.assertEqual(round(neuron([Var(1), Var(-0.5), Var(1.5)]).data), 1.0)
        self.assertEqual(round(neuron([Var(1), Var(-3), Var(1.5)]).data), -1.0)

        # test with sigmoid activation function
        neuron.activation = 'sigmoid'
        self.assertEqual(round(neuron([Var(1), Var(-0.5), Var(1.5)]).data), 1.0)
        self.assertEqual(round(neuron([Var(1), Var(-3), Var(1.5)]).data), 0.0)

    def test_layer(self):
        layer = Layer(in_features=3, out_features=2)
        self.assertEqual(len(layer.neurons), 2)
        self.assertEqual(len(layer.parameters()), 8)  # 3*2 + 2

        for neuron in layer.neurons:
            self.assertIsNone(neuron.activation)

        layer_with_activation = Layer(in_features=3, out_features=2, activation='relu')
        self.assertEqual(len(layer_with_activation.neurons), 2)
        self.assertEqual(len(layer_with_activation.parameters()), 8)  # 3*2 + 2
        for neuron in layer_with_activation.neurons:
            self.assertIs(neuron.activation, 'relu')

    def test_mlp(self):
        with self.assertRaises(AssertionError):
            incorrect_model_object_1 = MLP(in_features=2, layers=[3, 1], activations='relu')
            incorrect_model_object_2 = MLP(in_features=2, layers=[3, 1], activations=['relu'])

        model = MLP(in_features=2, layers=[3, 1], activations=['relu', 'linear'])  # 1 hidden layer with 3 neurons

        self.assertEqual(len(model.layers), 2)
        self.assertEqual(len(model.parameters()), 13)  # 2*3+3 + 3*1+1

        for layer, activation in zip(model.layers, ['relu', 'linear']):
            for neuron in layer.neurons:
                self.assertIs(neuron.activation, activation)


if __name__ == "__main__":
    unittest.main()
