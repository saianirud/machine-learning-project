import scipy.io
import numpy as np
import math

# calculates the overall, individual class accuracies and returns them as an array 
def calcAccuracy(data, res):
    
    # actual count of class 1
    count_class1 = np.count_nonzero(data)
    # actual count of class 0
    count_class0 = len(data) - count_class1
    # correctly predicted count overall
    count = 0
    # correctly predicted count of class 0
    count_y0 = 0
    # correctly predicted count of class 1
    count_y1 = 0
    
    for i in range(len(data)):
        if(res[i] == data[i]):
            count = count + 1
            if (res[i] == 0):
                count_y0 = count_y0 + 1
            else:
                count_y1 = count_y1 + 1
    return np.asarray([(count / (len(res))) * 100, (count_y0 / count_class0) * 100, (count_y1 / count_class1) * 100])

# loading the data
data = scipy.io.loadmat('fashion_mnist.mat')

# reducing 784 pixels of each image to 2 features for training dataset
# feature1 = mean and feature2 = standard deviation of the pixels
trX = data['trX']
trX_feature1 = np.mean(trX, 1)
trX_feature2 = np.std(trX, 1)

# reducing 784 pixels of each image to 2 features for testing dataset
# feature1 = mean and feature2 = standard deviation of the pixels
tsX = data['tsX']
tsX_feature1 = np.mean(tsX, 1)
tsX_feature2 = np.std(tsX, 1)

# reading label values
trY = data['trY']
tsY = data['tsY']

# preparing training set where each row represents a sample i.e. [feature1, feature2, label]
training_set = np.transpose(np.asarray([trX_feature1, trX_feature2, trY[0]]))

# preparing testing set where each row represents a sample i.e. [feature1, feature2, label]
testing_set = np.transpose(np.asarray([tsX_feature1, tsX_feature2, tsY[0]]))



# Naive Bayes

print('\n**********Naive Bayes**********\n')

# calculate the mean for both the features and return them as an array
def getMean(training_set):
    mean_feature1 = np.mean(training_set[:,0])
    mean_feature2 = np.mean(training_set[:,1])
    return np.asarray([mean_feature1, mean_feature2])

# calculate the standard deviation for both the features and return them as an array
def getStd(training_set):
    std_feature1 = np.std(training_set[:,0])
    std_feature2 = np.std(training_set[:,1])
    return np.asarray([std_feature1, std_feature2])

# returns the value of gaussian distribution for a given x, mean and standard deviation
def gaussianDistribution(x, mean, std):
    exp = math.exp(-0.5 * (math.pow((x-mean) / std, 2)))
    c = std * math.sqrt(2*math.pi)
    res = (1 / c) * exp
    return res

# seperating the training set according to the class
training_set_class0 = training_set[training_set[:,2] == 0]
training_set_class1 = training_set[training_set[:,2] == 1]

# calculating the mean and standard deviation of both the features in class 0
trX_class0_mean = getMean(training_set_class0) # this will return an array [mean_feature1, mean_feature2]
trX_class0_std = getStd(training_set_class0) # this will return an array [std_feature1, std_feature2]

print('Mean for Class 0: ', trX_class0_mean)
print('Standard Deviation for Class 0: ', trX_class0_std)

# calculating the mean and standard deviation of both the features in class 1
trX_class1_mean = getMean(training_set_class1)
trX_class1_std = getStd(training_set_class1)

print('Mean for Class 1: ', trX_class1_mean)
print('Standard Deviation for Class 1: ', trX_class1_std)

# prior probability of class 1
prob_class1 = np.count_nonzero(training_set[:,2]) / len(training_set[:,2])
# prior probability of class 0
prob_class0 = 1 - prob_class1

# predicted labels array for testing data
res_nb = []

# calculating p(x1/y) * p(x2/y) * p(y) for both the classes and predicting the label accordingly
for i in range(len(testing_set)):
    prob_x1_class0 = gaussianDistribution(testing_set[i][0], trX_class0_mean[0], trX_class0_std[0])
    prob_x2_class0 = gaussianDistribution(testing_set[i][1], trX_class0_mean[1], trX_class0_std[1])
    prob_class0_x = prob_x1_class0 * prob_x2_class0 * prob_class0

    prob_x1_class1 = gaussianDistribution(testing_set[i][0], trX_class1_mean[0], trX_class1_std[0])
    prob_x2_class1 = gaussianDistribution(testing_set[i][1], trX_class1_mean[1], trX_class1_std[1])
    prob_class1_x = prob_x1_class1 * prob_x2_class1 * prob_class1

    # predict the label as class 0 if p(x1/y) * p(x2/y) * p(y) is greater for y=0 otherwise predict it as class 1
    if (prob_class0_x > prob_class1_x):
        res_nb.append(0.0)
    else:
        res_nb.append(1.0)

# calculate the accuracies and print it
accuracy_nb = calcAccuracy(testing_set[:,2], res_nb)
print('Naive Bayes Accuracy Overall: {0}%'.format(accuracy_nb[0]))
print('Naive Bayes Accuracy for Y=0: {0}%'.format(accuracy_nb[1]))
print('Naive Bayes Accuracy for Y=1: {0}%'.format(accuracy_nb[2]))




# Logistic Regression

print('\n**********Logistic Regression**********\n')

# pass the dot product of feature vector and w vector to the sigmoid function and return the value
def sigmoidFunction(trX_feature, w):
    t = np.dot(trX_feature, w.T)
    return 1 / (1 + np.exp(-t))

# return the dot product of feature vector and the error vector i.e. x.(y-sigmoid(w.x))
def gradientAscent(trX_feature, trY, sigmoidY):
    gradient = np.dot(trX_feature.T, trY - sigmoidY)
    return gradient

# number of iterations
iterations = 10000
# learning rate
learning_rate = 0.01

print('Logistic Regression Learning rate: {0}'.format(learning_rate))
print('Logistic Regression Iterations: {0}'.format(iterations))

# modifying the training set such that each row represent [1, feature1, feature2, label]
# 1 is added because of the intercept
training_set = np.append(np.ones((training_set.shape[0], 1)), training_set, axis=1)

# initializing the w vector
w = np.zeros(training_set.shape[1]-1)

# gradient ascent
for i in range(iterations):
    # calculating p(y/x) by passing the dot product of feature vector and w vector to the sigmoid function
    sigY = sigmoidFunction(training_set[:, 0:3], w)
    # updating the w vector
    w = w + learning_rate * gradientAscent(training_set[:, 0:3], training_set[:,3], sigY)

# the final w parameters
print('Logistic Regression w parameters: {0}'.format(w))

# modifying the testing set such that each row represent [1, feature1, feature2, label]
testing_set = np.append(np.ones((testing_set.shape[0], 1)), testing_set, axis=1)

# calculating p(y/x)
sigY = sigmoidFunction(testing_set[:, 0:3], w)

# predicted labels array for testing data
res_lr = []

# predict the label as class 0 if p(y/x) < 0.5 otherwise predict it as class 1
for y in sigY:
    if y < 0.5:
        res_lr.append(0.0)
    else:
        res_lr.append(1.0)

# calculate the accuracy and print it
accuracy_lr = calcAccuracy(testing_set[:,3], res_lr)
print('Logistic Regression Accuracy Overall: {0}%'.format(accuracy_lr[0]))
print('Logistic Regression Accuracy for Y=0: {0}%'.format(accuracy_lr[1]))
print('Logistic Regression Accuracy for Y=1: {0}%'.format(accuracy_lr[2]))