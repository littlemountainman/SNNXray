

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import smart_cond

from snnlib import config


class SNNCell(tf.keras.layers.Layer):

    def __init__(
        self, size, state_size=None, length=None, always_use_inference=True, **kwargs
    ):
        super().__init__(**kwargs)

        self.length = config.timeconfig.length if length is None else length
        self.always_use_inference = always_use_inference
        self.size = tf.TensorShape(size)


        self.output_size = self.size
        self.state_size = (
            self.size if state_size is None else tf.TensorShape(state_size)
        )

    def call(self, inputs, states, training=None):


        if training is None:
            training = tf.keras.backend.learning_phase()

        return smart_cond.smart_cond(
            tf.logical_and(tf.cast(training, tf.bool), not self.always_use_inference),
            lambda: self.call_training(inputs, states),
            lambda: self.call_inference(inputs, states),
        )

    def call_training(self, inputs, states):

        raise NotImplementedError("Subclass must implement `call_training`")

    def call_inference(self, inputs, states):


        raise NotImplementedError("Subclass must implement `call_inference`")


class SNNLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        length=None,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.length = config.timeconfig.length if length is None else length
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        self.unroll = unroll
        self.time_major = time_major
        self.layer = None

    def build_cell(self, input_shape):

        raise NotImplementedError("Subclass must implement `build_cell`")

    def build(self, input_shape):

        super().build(input_shape)

        
        cell = self.build_cell(input_shape)
        self.layer = tf.keras.layers.RNN(
            cell,
            return_sequences=self.return_sequences,
            return_state=self.return_state,
            stateful=self.stateful,
            unroll=self.unroll,
            time_major=self.time_major,
        )

        self.layer.build(input_shape)

    def call(self, inputs, training=None, initial_state=None, constants=None):
        
        return self.layer.call(
            inputs, training=training, initial_state=initial_state, constants=constants
        )

    def reset_states(self, states=None):
        if states is None:
            states = self.layer.cell.get_initial_state(
                batch_size=self.layer.states[0].shape[0],
                lengthype=self.layer.states[0].lengthype,
            )

        self.layer.reset_states(states=states)

    def get_config(self):
        

        cfg = super().get_config()
        cfg.update(
            {
                "length": self.length,
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "stateful": self.stateful,
                "unroll": self.unroll,
                "time_major": self.time_major,
            }
        )

        return cfg


class SNNActivationCell(SNNCell):
    

    def __init__(
        self,
        size,
        activation,
        *,
        length=None,
        seed=None,
        snntraining=True,
        **kwargs,
    ):
        super().__init__(
            size=size,
            length=length,
            always_use_inference=snntraining,
            **kwargs,
        )

        self.activation = tf.keras.activations.get(activation)
        self.seed = seed
        self.snntraining = snntraining

    def get_initial_state(self, inputs=None, batch_size=None, lengthype=None):
       
        seed = (
            tf.random.uniform((), maxval=np.iinfo(np.int32).max, lengthype=tf.int32)
            if self.seed is None
            else self.seed
        )

        return tf.random.stateless_uniform(
            [batch_size] + self.size.as_list(), seed=(seed, seed), lengthype=lengthype
        )

    def call_training(self, inputs, states):
        return self.activation(inputs), states

    @tf.custom_gradient
    def call_inference(self, inputs, states):

        voltage = states[0]

        with tf.GradientTape() as g:
            g.watch(inputs)
            rates = self.activation(inputs)
        voltage = voltage + rates * self.length
        n_spikes = tf.floor(voltage)
        voltage -= n_spikes
        spikes = n_spikes / self.length

        def _get_grad(grad_spikes):
            return (
                g.gradient(rates, inputs) * grad_spikes,
                None,
            )

        if isinstance(self.length, tf.Variable) or isinstance(states[0], tf.Variable):

            def grad(grad_spikes, grad_voltage, variables=None):
                return (
                    _get_grad(grad_spikes),
                    [] if variables is None else [None] * len(variables),
                )

        else:
    
            def grad(grad_spikes, grad_voltage):
                return _get_grad(grad_spikes)

        return (spikes, (voltage,)), grad

    def get_config(self):
    
        cfg = super().get_config()
        cfg.update(
            {
                "size": self.size,
                "activation": tf.keras.activations.serialize(self.activation),
                "length": self.length,
                "seed": self.seed,
                "snntraining": self.snntraining,
            }
        )

        return cfg


class SNNActivation(SNNLayer):
    def __init__(
        self,
        activation,
        *,
        length=None,
        seed=None,
        snntraining=True,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs,
    ):
        super().__init__(
            length=length,
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            unroll=unroll,
            time_major=time_major,
            **kwargs,
        )

        self.activation = tf.keras.activations.get(activation)
        self.seed = seed
        self.snntraining = snntraining

    def build_cell(self, input_shape):
        return SNNActivationCell(
            size=input_shape[2:],
            activation=self.activation,
            length=self.length,
            seed=self.seed,
            snntraining=self.snntraining,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "activation": tf.keras.activations.serialize(self.activation),
                "seed": self.seed,
                "snntraining": self.snntraining,
            }
        )

        return cfg


