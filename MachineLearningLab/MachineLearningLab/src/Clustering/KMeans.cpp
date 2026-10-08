#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  KMeans class implementation  ///

// KMeans function: Constructor for KMeans class.//
KMeans::KMeans(int numClusters, int maxIterations)
	: numClusters_(numClusters), maxIterations_(maxIterations) {}


// fit function: Performs K-means clustering on the given dataset and return the centroids of the clusters.//
void KMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;

	int numPoints = normalizedData.size();
	int numFeatures = normalizedData[0].size();

	/* --- Initialize centroids randomly --- */
	centroids_.clear();
	std::vector<int> indices(numPoints);

	for (int i = 0; i < numPoints; ++i) {
		indices[i] = i;
	}

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(indices.begin(), indices.end(), g);

	/* --- Randomly select unique centroid indices --- */
	for (int i = 0; i < numClusters_; ++i) {
		centroids_.push_back(normalizedData[indices[i]]);
	}

	/* --- Perform K-means clustering --- */
	for (int iter = 0; iter < maxIterations_; ++iter) {

		/* --- Assign data points to the nearest centroid --- */
		std::vector<int> labels = predict(normalizedData);

		std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(numFeatures, 0.0));
		std::vector<int> clusterCounts(numClusters_, 0);

		/* --- Update newCentroids and clusterCounts --- */
		for (int i = 0; i < numPoints; ++i) {
			int clusterIdx = labels[i];
			for (int j = 0; j < numFeatures; ++j) {
				newCentroids[clusterIdx][j] += normalizedData[i][j];
			}
			clusterCounts[clusterIdx]++;
		}

		bool converged = true;

		/* --- Update centroids --- */
		for (int c = 0; c < numClusters_; ++c) {
			if (clusterCounts[c] > 0) {
				for (int j = 0; j < numFeatures; ++j) {
					newCentroids[c][j] /= clusterCounts[c];
				}
			}
			else {
				// Evitar que un centroide se quede sin puntos asignados
				newCentroids[c] = centroids_[c];
			}

			/* --- Check for convergence --- */
			for (int j = 0; j < numFeatures; ++j) {
				// Si la diferencia entre el centroide viejo y el nuevo es significativa, no ha convergido
				if (std::abs(centroids_[c][j] - newCentroids[c][j]) > 1e-6) {
					converged = false;
					break;
				}
			}
		}

		centroids_ = newCentroids;

		if (converged) {
			break; // Sale del bucle si los centroides ya no se mueven
		}
	}
}


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size());
	
	for (const auto& point : data) {
		/* --- Initialize the closest centroid and minimum distance to the maximum possible value --- */
		double minDistance = (std::numeric_limits<double>::max)();
		int closestCentroid = -1;

		/* --- Iterate through each centroid --- */
		for (int c = 0; c < numClusters_; ++c) {

			/* --- Calculate the Euclidean distance between the point and the centroid --- */
			double dist = SimilarityFunctions::euclideanDistance(point, centroids_[c]);

			if (dist < minDistance) {
				minDistance = dist;
				closestCentroid = c;
			}
		}

		/* --- Add the closest centroid to the labels vector --- */
		labels.push_back(closestCentroid);
	}

	return labels; // Return the labels vector

}





/// runKMeans: this function runs the KMeans clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
KMeans::runKMeans(const std::string& filePath) {
	DataPreprocessor DataPreprocessor;
	try {
		// Check if the file path is empty
		if (filePath.empty()) {
			MessageBox::Show("Please browse and select the dataset file from your PC.");
			return {}; // Return an empty vector since there's no valid file path
		}

		// Attempt to open the file
		std::ifstream file(filePath);
		if (!file.is_open()) {
			MessageBox::Show("Failed to open the dataset file");
			return {}; // Return an empty vector since file couldn't be opened
		}

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);

		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		MessageBox::Show("Run completed");
		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	}
}