import numpy
# scipy.special for the sigmoid function expit()
from scipy import special


class mynn:
    
    # initialise the neural network
    def __init__(self, inputneurons, hiddenneurons, outputneurons, learningrate):
        # set number of neurons in each input, hidden, output layer
        self.ineurons = inputneurons
        self.hneurons = hiddenneurons
        self.oneurons = outputneurons
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.ineurons, -0.5), (self.hneurons, self.ineurons))
        self.who = numpy.random.normal(0.0, pow(self.hneurons, -0.5), (self.oneurons, self.hneurons))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden neurons
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
# number of input, hidden and output neurons
input_neurons = 784
hidden_neurons = 200
output_neurons = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = mynn(input_neurons,hidden_neurons,output_neurons, learning_rate)
# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()[1:]
training_data_file.close()

# train the neural network
print("Begin training")
# epochs is the number of times the training data set is used for training
epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_neurons) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# load the mnist verify data CSV file into a list
print("*******************************")
print("Begin verifying")
verify_data_file = open("mnist_dataset/mnist_verify.csv", 'r')
verify_data_list = verify_data_file.readlines()[1:]
verify_data_file.close()

# verify the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the verify data set
for record in verify_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)


test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()[1:]
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
label_of_test = []
print("*******************************")
print("Begin testing")
# go through all the records in the test data set
for record in test_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # append label to list
    label_of_test.append(numpy.argmax(outputs))
    pass

print("*******************************")
print("Output labels to file")
# write labels to text file
with open("label_test.txt","w") as f_test:
    for label in label_of_test:
        f_test.write(str(label))
        f_test.write("\n")
