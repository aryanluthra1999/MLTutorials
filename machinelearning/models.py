import nn
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """

        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        num = nn.as_scalar(self.run(x))
        if num >= 0:
            return 1
        else:
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        alpha = 0.05

        batch_size = 1
        while(True):
            num_mistakes = 0
            for x, y in dataset.iterate_once(batch_size):
                result = self.get_prediction(x)
                if nn.as_scalar(y) != result:
                    self.get_weights().update(x, nn.as_scalar(y))
                    num_mistakes += 1

            if num_mistakes == 0:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        l1_size = 55
        self.l1_weights_in = nn.Parameter(1, l1_size)
        self.l1_bias_in = nn.Parameter(1, l1_size)

        self.l1_weights_out = nn.Parameter(l1_size, 1)
        self.l1_bias_out = nn.Parameter(1, 1)

        self.learning_rate = -0.005



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        in_nodes = nn.Linear(x, self.l1_weights_in)
        in_nodes = nn.AddBias(in_nodes, self.l1_bias_in)
        in_nodes = nn.ReLU(in_nodes)
        in_nodes = nn.Linear(in_nodes, self.l1_weights_out)
        in_nodes = nn.AddBias(in_nodes, self.l1_bias_out)
        return in_nodes

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss_node = nn.SquareLoss(self.run(x), y)
        return loss_node


    def train(self, dataset):
        """
        Trains the model.
        """

        avg_loss = 1
        batch_size = 10

        while (avg_loss >= 0.001):
            losses = []

            for x, y in dataset.iterate_once(batch_size):
                curr_loss = self.get_loss(x, y)

                grd_w_in, grd_b_in, grd_w_out, grd_b_out = nn.gradients(curr_loss, [self.l1_weights_in, self.l1_bias_in, self.l1_weights_out, self.l1_bias_out])

                self.l1_weights_in.update(grd_w_in, self.learning_rate)
                self.l1_bias_in.update(grd_b_in, self.learning_rate)

                self.l1_weights_out.update(grd_w_out, self.learning_rate)
                self.l1_bias_out.update(grd_b_out, self.learning_rate)

                losses.append(nn.as_scalar(curr_loss))

            avg_loss = np.mean(losses)



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        l1_size = 100
        self.l1_weights_in = nn.Parameter(784, l1_size)
        self.l1_bias_in = nn.Parameter(1, l1_size)

        self.l1_weights_out = nn.Parameter(l1_size, 10)
        self.l1_bias_out = nn.Parameter(1, 10)

        self.learning_rate = -0.01


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        in_nodes = nn.Linear(x, self.l1_weights_in)
        in_nodes = nn.AddBias(in_nodes, self.l1_bias_in)
        in_nodes = nn.ReLU(in_nodes)
        in_nodes = nn.Linear(in_nodes, self.l1_weights_out)
        in_nodes = nn.AddBias(in_nodes, self.l1_bias_out)

        return in_nodes


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss_node = nn.SoftmaxLoss(self.run(x), y)
        return loss_node

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        val_acc = 0
        batch_size = 20

        while (val_acc < 0.98):
            for x, y in dataset.iterate_once(batch_size):
                curr_loss = self.get_loss(x, y)

                grd_w_in, grd_b_in, grd_w_out, grd_b_out = nn.gradients(curr_loss, [self.l1_weights_in, self.l1_bias_in,
                                                                                    self.l1_weights_out,
                                                                                    self.l1_bias_out])

                self.l1_weights_in.update(grd_w_in, self.learning_rate)
                self.l1_bias_in.update(grd_b_in, self.learning_rate)

                self.l1_weights_out.update(grd_w_out, self.learning_rate)
                self.l1_bias_out.update(grd_b_out, self.learning_rate)

            val_acc = dataset.get_validation_accuracy()

            #if val_acc >= 0.965:
            #    self.learning_rate = self.learning_rate/2


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        l1_size = 256
        self.w_in= nn.Parameter(self.num_chars, l1_size)
        self.b_in = nn.Parameter(1, l1_size)

        self.w_hidden = nn.Parameter(l1_size, l1_size)
        self.b_hidden = nn.Parameter(1, l1_size)

        self.w_out = nn.Parameter(l1_size, 5)
        self.b_out = nn.Parameter(1, 5)

        self.learning_rate = -0.05



    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        inputs = None
        for char in xs:
            if inputs == None:
                inputs = nn.AddBias(nn.Linear(char, self.w_in), self.b_in)
                inputs = nn.ReLU(inputs)

            inputs = nn.Add(nn.AddBias(nn.Linear(char, self.w_in), self.b_in), nn.Linear(inputs, self.w_hidden))
            inputs = nn.ReLU(nn.AddBias(inputs, self.b_hidden))

        inputs = nn.Linear(inputs, self.w_out)
        inputs = nn.AddBias(inputs, self.b_out)
        return inputs


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        val_acc = 0
        batch_size = 200

        while val_acc <= 0.86:
            for x, y in dataset.iterate_once(batch_size):
                curr_loss = self.get_loss(x, y)

                grd_w_in, grd_b_in, grd_w_out, grd_b_out, grd_b_hidden, grd_w_hidden = nn.gradients(curr_loss, [self.w_in, self.b_in, self.w_out, self.b_out, self.b_hidden, self.w_hidden])

                self.w_in.update(grd_w_in, self.learning_rate)
                self.w_hidden.update(grd_w_hidden, self.learning_rate)
                self.b_in.update(grd_b_in, self.learning_rate)
                self.b_hidden.update(grd_b_hidden, self.learning_rate)

                self.w_out.update(grd_w_out, self.learning_rate)
                self.b_out.update(grd_b_out, self.learning_rate)

                val_acc = dataset.get_validation_accuracy()
                if val_acc >= 86:
                    break

            # if val_acc >= 0.965:
            #    self.learning_rate = self.learning_rate/2

