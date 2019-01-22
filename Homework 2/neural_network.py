from __future__ import division,print_function

import numpy as np

class Network(object):
    """
    Neural network for softmax regression problems
    """

    def __init__(self,layer_number_of_nodes,layer_activation_functions,layer_has_bias,layer_weight_means_and_stds=None, gama=0.1, regularization=None):
        self.layer_number_of_nodes = layer_number_of_nodes           # Of nodes in each layer
        self.layer_activation_functions = [None]
        # Add an identity activation function here!
        for act in layer_activation_functions:
            if act=='softmax':
                self.layer_activation_functions.append(self._softmax) 
            if act=='sigmoid':
                self.layer_activation_functions.append(self._sigmoid) 
            if act=='leaky_relu':
                self.layer_activation_functions.append(self._leaky_relu) 
            if act=='gaussian':
                self.layer_activation_functions.append(self._gaussian) 
            if act=='identity':
                self.layer_activation_functions.append(self._identity)

        self.layer_has_bias = layer_has_bias                         # Whether to add a bias node to each layer
        self.L = len(self.layer_number_of_nodes)                     # Number of layers
        self.gama = gama

        self.weights = [np.array([])]
        self.regularization = regularization

        # Create arrays to hold the weights, which are N_l(+1) by N_(l+1)
        for i in range(self.L-1):
            # if we have a normal distribution and standard deviation, then generate random weights from that distribution,
            if layer_weight_means_and_stds is not None:
                w = layer_weight_means_and_stds[i][1]*np.random.randn(self.layer_number_of_nodes[i] + self.layer_has_bias[i],self.layer_number_of_nodes[i+1]) + layer_weight_means_and_stds[i][0] 
            # Otherwise just initialize the weights to zero
            else:
                w = np.zeros((self.layer_number_of_nodes[i] + self.layer_has_bias[i],self.layer_number_of_nodes[i+1]))
            self.weights.append(w)
  
    def feed_forward(self,feature):
        # evaluate the neural network for a vector-valued input
        m = feature.shape[0]

        # Append a column of ones to the input if a bias is desired
        if self.layer_has_bias[0]:
            z = np.column_stack((np.ones((m)),feature))
        else:
            z = feature

        # Initialize lists to hold the node inputs and outputs, treating the input values as the output of the first node
        self.a_vals = [None]
        self.z_vals = [z]

        # Loop over the remaining layers
        for l in range(1,self.L):
            # Take the linear combination of the previous layers outputs (z^(l-1)) and weights (w^(l)) to form a^(l)
            a = np.dot(self.z_vals[l-1],self.weights[l])
            # Run a through the activation function to form z^(l)
            z = self.layer_activation_functions[l](a) 
            # If a bias is desired, append a column of ones to z
            if self.layer_has_bias[l]:         
                z = np.column_stack((np.ones((m)),z))
            # Store these values (for computing the gradient later)
            self.a_vals.append(a) 
            self.z_vals.append(z)
        return z

    def _J_fun(self,feature,label):
        # Add your sum square error evaluation here!
        if self.layer_activation_functions[-1]==self._identity:
            cost_function_data = np.sum(np.sum((1/2) * np.power(label - self.feed_forward(feature), 2), axis=1), axis=0)
        if self.layer_activation_functions[-1]==self._softmax:
            # Model objective function -- Cross-entropy 
             cost_function_data = -np.sum(np.sum(label*np.log(self.feed_forward(feature)),axis=1),axis=0)
        else:
            print('Only softmax supported for final layer')            

        # Add regularization here!
        # TODO
        cost_function_reg = 0
        if(self.regularization == 'L1'):
            for w in self.weights:
                cost_function_reg = self.gama * np.sum(np.abs(w))
        if(self.regularization == 'L2'):
            for w in self.weights:
                cost_function_reg = self.gama * np.sum(np.power(w, 2))

        return cost_function_data + cost_function_reg

    def _gradient_fun(self,feature,label):
        # Compute the gradient via backpropagation
        m = feature.shape[0]

        # Initialize gradient arrays (same shape as the weights)
        grads = [np.zeros_like(w) for w in self.weights]

        # Compute dJ/da (aka the delta term) for the final layer.  This often involves 
        # Some algebraic simplification when cost function is selected judiciously, so
        # this is coded by hand here.

        l = self.L-1 #last layer

        z = self.z_vals[l]              # Current layer out
        z_previous = self.z_vals[l-1]   # Last layer out
        a = self.a_vals[l]              # Current layer in
        w = self.weights[l]             # Last layer weights
        activation = self.layer_activation_functions[l]     #Current layer activation
        if activation==self._softmax or activation==self._identity:
            delta_l = (z - label)                       # Current layer error
        # Add gradient of SSE here!
        else:
            print('Only softmax and identity supported for final layer') 

        grads[l] = np.dot(z_previous.T,delta_l)    # gradient due to data misfit

        # Add gradient of regularization here!
        model_norm_gradient = 0
        if(self.regularization == 'L1'):
            model_norm_gradient = self.gama * np.sign(w)
        if(self.regularization == 'L2'):
            model_norm_gradient = self.gama * w
        grads[l] += model_norm_gradient                    # add gradient due to regularization

        # Loop over the remaining layers
        for l in range(self.L-2,0,-1):
        
            z_previous = self.z_vals[l-1]                    # last layer output
            a = self.a_vals[l]                                # Current layer input

            w_next = self.weights[l+1][1:] # weights from the next layer, excluding bias weights
            activation = self.layer_activation_functions[l]  # Current layer activation

            delta_l = np.dot(delta_l,w_next.T)*activation(a,dx=1)  # Current layer error
            grads[l] = np.dot(z_previous.T,delta_l)  # Gradient due to data misfit

            # Add gradient of regularization here!
            model_norm_gradient = 0
            if(self.regularization == 'L1'):
                model_norm_gradient = self.gama * np.sign(self.weights[l])
            if(self.regularization == 'L2'):
                model_norm_gradient = self.gama * self.weights[l]
            grads[l] += model_norm_gradient             # add gradient due to regularization

        return grads

    @staticmethod
    def _softmax(X,dx=0):
        if dx==0:
            return np.exp(X)/np.repeat(np.sum(np.exp(X),axis=1,keepdims=True),X.shape[1],axis=1)
    @staticmethod
    def _sigmoid(X,dx=0):
        if dx==0:
            return 1./(1+np.exp(-X))
        if dx==1:
            s = 1./(1+np.exp(-X))
            return s*(1-s)

    @staticmethod
    def _leaky_relu(X,dx=0):
        if dx==0:
            return (X>0)*X + 0.01*(X<=0)*X
        if dx==1:
            return (X>0) + 0.01*(X<=0)

    @staticmethod
    def _gaussian(X,dx=0):
        if dx==0:
            return np.exp(-X**2)
        if dx==1:
            return -2*X*np.exp(-X**2)

    # Add @staticmethod for identity here
    @staticmethod
    def _identity(X,dx=0):
        if dx == 0:
            return X 
        return np.onelike(X)
            