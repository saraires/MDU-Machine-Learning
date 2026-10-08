#include "FuzzyCMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  FuzzyCMeans class implementation  ///


// FuzzyCMeans function: Constructor for FuzzyCMeans class.//
FuzzyCMeans::FuzzyCMeans(int numClusters, int maxIterations, double fuzziness)
	: numClusters_(numClusters), maxIterations_(maxIterations), fuzziness_(fuzziness) {}


// fit function: Performs Fuzzy C-Means clustering on the given dataset and return the centroids of the clusters.//
void FuzzyCMeans::fit(const std::vector<std::vector<double>>& data) {
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

	for (int i = 0; i < numClusters_; ++i) {
		centroids_.push_back(normalizedData[indices[i]]);
	}

	/* --- Initialize the membership matrix with the number of data points --- */
	initializeMembershipMatrix(numPoints);

	// Actualizamos inicialmente la matriz con los centroides aleatorios generados
	updateMembershipMatrix(normalizedData, centroids_);

	/* --- Perform Fuzzy C-means clustering --- */
	for (int iter = 0; iter < maxIterations_; ++iter) {
		std::vector<std::vector<double>> oldCentroids = centroids_;

		// 1. Actualizar centroides basándose en las pertenencias actuales
		updateCentroids(normalizedData);

		// 2. Actualizar las pertenencias usando los nuevos centroides
		updateMembershipMatrix(normalizedData, centroids_);

		// 3. Comprobar convergencia (si los centroides ya no se mueven)
		bool converged = true;
		for (int j = 0; j < numClusters_; ++j) {
			for (int f = 0; f < numFeatures; ++f) {
				if (std::abs(oldCentroids[j][f] - centroids_[j][f]) > 1e-5) {
					converged = false;
					break;
				}
			}
			if (!converged) break;
		}

		if (converged) break;
	}
}


// initializeMembershipMatrix function: Initializes the membership matrix with random values that sum up to 1 for each data point.//
void FuzzyCMeans::initializeMembershipMatrix(int numDataPoints) {
	membershipMatrix_.clear();
	membershipMatrix_.resize(numDataPoints, std::vector<double>(numClusters_, 0.0));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.01, 1.0); // Evitar ceros exactos

	/* --- Initialize membership matrix with random values that sum up to 1 for each data point --- */
	for (int i = 0; i < numDataPoints; ++i) {
		double sum = 0.0;
		for (int j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] = dis(gen);
			sum += membershipMatrix_[i][j];
		}

		/* --- Normalize membership values to sum up to 1 for each data point --- */
		for (int j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] /= sum;
		}
	}
}


// updateMembershipMatrix function: Updates the membership matrix using the fuzzy c-means algorithm.//
void FuzzyCMeans::updateMembershipMatrix(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>> centroids_) {

	int numPoints = data.size();
	double p = 2.0 / (fuzziness_ - 1.0); // Exponente de la fórmula FCM

	/* --- Iterate through each data point --- */
	for (int i = 0; i < numPoints; ++i) {
		std::vector<double> distances(numClusters_);
		bool pointOnCentroid = false;
		int exactCentroidIdx = -1;

		for (int j = 0; j < numClusters_; ++j) {
			/* --- Calculate the distance between the data point and the centroid --- */
			distances[j] = SimilarityFunctions::euclideanDistance(data[i], centroids_[j]);

			if (distances[j] < 1e-10) { // Si el punto es idéntico al centroide
				pointOnCentroid = true;
				exactCentroidIdx = j;
			}
		}

		/* --- Update the membership matrix with the new value --- */
		if (pointOnCentroid) {
			for (int j = 0; j < numClusters_; ++j) {
				membershipMatrix_[i][j] = (j == exactCentroidIdx) ? 1.0 : 0.0;
			}
		}
		else {
			double totalSum = 0.0;
			for (int j = 0; j < numClusters_; ++j) {
				double denomSum = 0.0;
				for (int k = 0; k < numClusters_; ++k) {
					denomSum += std::pow(distances[j] / distances[k], p);
				}
				membershipMatrix_[i][j] = 1.0 / denomSum;
				totalSum += membershipMatrix_[i][j];
			}

			/* --- Normalize membership values to sum up to 1 for each data point --- */
			for (int j = 0; j < numClusters_; ++j) {
				membershipMatrix_[i][j] /= totalSum;
			}
		}
	}
}


// updateCentroids function: Updates the centroids of the Fuzzy C-Means algorithm.//
std::vector<std::vector<double>> FuzzyCMeans::updateCentroids(const std::vector<std::vector<double>>& data) {
	int numPoints = data.size();
	int numFeatures = data[0].size();
	std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(numFeatures, 0.0));

	/* --- Iterate through each cluster --- */
	for (int j = 0; j < numClusters_; ++j) {
		double denominator = 0.0;

		/* --- Iterate through each data point --- */
		for (int i = 0; i < numPoints; ++i) {

			/* --- Calculate the membership of the data point to the cluster raised to the fuzziness --- */
			double u_ij_m = std::pow(membershipMatrix_[i][j], fuzziness_);
			denominator += u_ij_m;

			for (int f = 0; f < numFeatures; ++f) {
				newCentroids[j][f] += u_ij_m * data[i][f];
			}
		}

		if (denominator > 0) {
			for (int f = 0; f < numFeatures; ++f) {
				newCentroids[j][f] /= denominator;
			}
		}
		else {
			newCentroids[j] = centroids_[j];
		}
	}

	centroids_ = newCentroids;
	return centroids_; // Return the centroids
}

// predict function: Predicts the cluster labels for the given data points using the Fuzzy C-Means algorithm.//
std::vector<int> FuzzyCMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels; // Create a vector to store the labels
	labels.reserve(data.size()); // Reserve space for the labels

	double p = 2.0 / (fuzziness_ - 1.0);

	/* --- Iterate through each point in the data --- */
	for (const auto& point : data) {
		int closestCentroid = -1;
		double maxMembership = -1.0;

		std::vector<double> distances(numClusters_);
		bool pointOnCentroid = false;
		int exactCentroidIdx = -1;

		/* --- Iterate through each centroid --- */
		for (int c = 0; c < numClusters_; ++c) {
			/* --- Calculate the distance between the point and the centroid --- */
			distances[c] = SimilarityFunctions::euclideanDistance(point, centroids_[c]);
			if (distances[c] < 1e-10) {
				pointOnCentroid = true;
				exactCentroidIdx = c;
			}
		}

		/* --- Calculate the membership of the point to the centroid --- */
		if (pointOnCentroid) {
			closestCentroid = exactCentroidIdx;
		}
		else {
			for (int j = 0; j < numClusters_; ++j) {
				double denomSum = 0.0;
				for (int k = 0; k < numClusters_; ++k) {
					denomSum += std::pow(distances[j] / distances[k], p);
				}
				double membership = 1.0 / denomSum;

				if (membership > maxMembership) {
					maxMembership = membership;
					closestCentroid = j;
				}
			}
		}

		/* --- Add the label of the closest centroid to the labels vector --- */
		labels.push_back(closestCentroid);
	}

	return labels;
}


/// runFuzzyCMeans: this function runs the Fuzzy C-Means clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
FuzzyCMeans::runFuzzyCMeans(const std::string& filePath) {
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