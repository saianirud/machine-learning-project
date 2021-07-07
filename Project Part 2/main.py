import scipy.io
import numpy as np
import matplotlib.pyplot as plt 
import random

# loading the data
data = scipy.io.loadmat('AllSamples.mat')
dataset = data['AllSamples']


# K-means algorithm
def kMeansAlgo(k, dataset, centroids):
    cluster_centers = centroids.copy()
    
    # run the loop until the cluster centers converge
    while True:
        cluster_data = {}
        # cluster_data[i] ---> samples belonging to cluster i+1
        for i in range(k):
            cluster_data[i] = []
        
        # loop through the dataset
        for i in range(len(dataset)):
            # for each sample in the dataset:

            # calculate the euclidean distance between the sample and the cluster centers
            norms = [np.linalg.norm(dataset[i] - center) for center in cluster_centers]
            # get the nearest cluster center to the sample and assign the sample to that cluster
            min_index = norms.index(min(norms))
            cluster_data[min_index].append(dataset[i])

        prev_cluster_centers = cluster_centers.copy()
        converged = True
        
        # Once all the samples are assigned to clusters
        # Calculate the mean of each cluster and assign it as the center of that cluster
        for i in range(k):
            if len(cluster_data[i]) != 0:
                cluster_centers[i] = np.mean(cluster_data[i], 0)

        # Break the loop if the centers converge i.e. new cluster centers = previous cluster centers
        for i in range(len(cluster_centers)):
            diff = np.sum(cluster_centers[i] - prev_cluster_centers[i])
            if diff != 0.0:
                converged = False
                break
        
        if converged:
            break
    
    return cluster_centers, cluster_data


# apply K-means strategy 1 on the given dataset
def kMeansStrategy1(k, dataset):
    # get the intial cluster centers randomly from the given dataset
    cluster_centers = np.asarray(random.choices(dataset, k = k))
    # Once the initial cluster centers are calculated, run K-means algorithm on the dataset
    final_centers, cluster_data = kMeansAlgo(k, dataset, cluster_centers)

    return final_centers, cluster_data


# apply K-means strategy 2 on the given dataset
def kMeansStrategy2(k, dataset):
    # pick the first center randomly from the dataset
    first_center_index = random.randrange(len(dataset))
    first_center = dataset[first_center_index]
    # remove the first center from the dataset
    dataset_modified = np.delete(dataset, first_center_index, 0)
    cluster_centers = [first_center]

    # calculate the remaining cluster centers
    for _ in range(k-1):
        max = 0
        max_index = -1

        # loop through the remaining dataset
        for i in range(len(dataset_modified)):
            # for each sample in the remaining dataset:

            # calculate the average of the euclidean distance between the sample and the previous cluster centers
            norms = [np.linalg.norm(dataset_modified[i] - center) for center in cluster_centers]
            avg = np.mean(norms)
            if avg > max:
                max = avg
                max_index = i
        
        # assign the sample with maximum average as a cluster center
        cluster_centers.append(dataset_modified[max_index])
        # remove the sample from the dataset
        dataset_modified = np.delete(dataset_modified, max_index, 0)

    cluster_centers = np.asarray(cluster_centers)
    # Once the initial cluster centers are calculated, run K-means algorithm on the dataset
    final_centers, cluster_data = kMeansAlgo(k, dataset, cluster_centers)

    return final_centers, cluster_data


# calculate the objective function sum
def objectiveFunctionSum(final_centers, cluster_data):
    objective_sum = 0

    # loop through all clusters
    for i in range(len(final_centers)):
        # for each cluster:

        # get the euclidean distance between the center and the data samples belonging to that cluster
        # final_centers[i] ---> final cluster center of cluster i+1
        # cluster_data[i] ---> samples belonging to cluster i+1
        norms = [np.linalg.norm(data - final_centers[i]) for data in cluster_data[i]]
        # square the distances calculated
        squared_norms = np.square(norms)
        # add to the objective_sum
        objective_sum = objective_sum + np.sum(squared_norms)
    
    return objective_sum


# plot the graph with given x values and y values
def plotGraph(n, x, y, xlabel, ylabel, title):
    plt.figure(n)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)



kValues = range(2, 11)


# 2 loops for 2 different initializations
for i in range(0, 2):


    # K-Means Strategy 1

    # to store the objectives values to the corresponding k
    objective_values = []

    print('\n******K - Means Strategy 1 Initialization {0}******'.format(i+1))
    for k in kValues:
        # get the final cluster centers and the cluster sets containing samples using strategy 1
        # final_centers[i] ---> final cluster center of cluster i+1
        # cluster_data[i] ---> samples belonging to cluster i+1
        final_centers, cluster_data = kMeansStrategy1(k, dataset)
        # calculate the objective function sum
        objective_sum = objectiveFunctionSum(final_centers, cluster_data)
        # append to objective_values
        objective_values.append([k, objective_sum])

    objective_values = np.asarray(objective_values)
    print('Objective Values:\n', objective_values)
    # plot the objective function vs k graph
    plotGraph(2*i+1, objective_values[:,0], objective_values[:,1], 'Number of Clusters k', 'Objective Function', 'K - Means Strategy 1 Initialization {0}'.format(i+1))


    # K-Means Strategy 2

    # to store the objectives values to the corresponding k
    objective_values = []

    print('\n******K - Means Strategy 2 Initialization {0}******'.format(i+1))
    for k in kValues:
        # get the final cluster centers and the cluster sets containing samples using strategy 2
        # final_centers[i] ---> final cluster center of cluster i+1
        # cluster_data[i] ---> samples belonging to cluster i+1
        final_centers, cluster_data = kMeansStrategy2(k, dataset)
        # calculate the objective function sum
        objective_sum = objectiveFunctionSum(final_centers, cluster_data)
        # append to objective_values
        objective_values.append([k, objective_sum])

    objective_values = np.asarray(objective_values)
    print('Objective Values:\n', objective_values)
    # plot the objective function vs k graph
    plotGraph(2*i+2, objective_values[:,0], objective_values[:,1], 'Number of Clusters k', 'Objective Function', 'K - Means Strategy 2 Initialization {0}'.format(i+1))


# show all graphs
plt.show()