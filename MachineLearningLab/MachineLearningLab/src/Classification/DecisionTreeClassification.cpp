#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	
	// Por cada flor, la bajamos por el árbol y guardamos su predicción
	for (int i = 0; i < (int)X.size(); i++) {
		double pred = traverseTree(X[i], root);
		predictions.push_back(pred);
	}
	
	return predictions;
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	
	int n_samples = X.size();
	int n_labels = std::set<double>(y.begin(), y.end()).size();  // cuántas especies distintas hay

	// --- define stopping criteria
	if (depth >= max_depth || n_labels == 1 || n_samples < min_samples_split) {
		double leaf_value = mostCommonlLabel(y);
		return new Node(0, 0.0, nullptr, nullptr, leaf_value);
	}

	double best_gain = -1.0; // set the best gain to -1
	int split_idx = NULL; // split index
	double split_thresh = NULL; // split threshold
	
	// --- Loop through candidate features and potential split thresholds
	for (int feat = 0; feat < n_feats; feat++) {
		std::vector<double> X_column;
		for (int i = 0; i < n_samples; i++) {
			X_column.push_back(X[i][feat]);        // saco la columna de esa medida
		}
		for (double threshold : X_column) {        // pruebo cada valor como umbral
			double gain = informationGain(y, X_column, threshold);
			if (gain > best_gain) {                // ¿es la mejor pregunta hasta ahora?
				best_gain = gain;
				split_idx = feat;
				split_thresh = threshold;
			}
		}
	}

	// --- si ninguna pregunta mejora nada, también hacemos HOJA
	if (best_gain <= 0.0) {
		double leaf_value = mostCommonlLabel(y);
		return new Node(0, 0.0, nullptr, nullptr, leaf_value);
	}

	// --- partir los datos según la mejor pregunta encontrada
	std::vector<std::vector<double>> X_left, X_right;
	std::vector<double> y_left, y_right;
	for (int i = 0; i < n_samples; i++) {
		if (X[i][split_idx] <= split_thresh) {
			X_left.push_back(X[i]);
			y_left.push_back(y[i]);
		}
		else {
			X_right.push_back(X[i]);
			y_right.push_back(y[i]);
		}
	}
	
	Node* left = growTree(X_left, y_left, depth + 1);
	Node* right = growTree(X_right, y_right, depth + 1);
	return new Node(split_idx, split_thresh, left, right);
	
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	double parent_entropy = EntropyFunctions::entropy(y);
	double ig = 0.0;
	
	// -- - generate split : separar las posiciones en izquierda y derecha
	std::vector<int> left_idxs, right_idxs;
	for (int i = 0; i < (int)X_column.size(); i++) {
		if (X_column[i] <= split_thresh)
			left_idxs.push_back(i);
		else
			right_idxs.push_back(i);
	}

	// si un lado queda vacío, esta pregunta no separa nada → ganancia 0
	if (left_idxs.empty() || right_idxs.empty())
		return 0.0;
	
	// --- compute the weighted avg. of the loss for the children
	int n = y.size();
	int n_left = left_idxs.size();
	int n_right = right_idxs.size();
	double e_left = EntropyFunctions::entropy(y, left_idxs);
	double e_right = EntropyFunctions::entropy(y, right_idxs);
	double child_entropy = ((double)n_left / n) * e_left + ((double)n_right / n) * e_right;

	// --- information gain is difference in loss before vs. after split
	ig = parent_entropy - child_entropy;

	return ig;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {	
	double most_common = 0.0;
	
	std::unordered_map<double, int> counts;
	for (double label : y) {
		counts[label]++;              // count the occurrences of each label	
	}
	int best_count = -1;
	for (const auto& par : counts) {  // stay with the label that has the highest count
		if (par.second > best_count) {
			best_count = par.second;
			most_common = par.first;
		}
	}
	return most_common;
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

	// Si el nodo es una hoja, devolvemos su valor (la especie)
	if (node->isLeafNode()) {
		return node->value;
	}

	// Si la medida de la flor es <= umbral, bajamos por la izquierda...
	if (x[node->feature] <= node->threshold) {
		return traverseTree(x, node->left);
	}
	// ...si no, por la derecha
	return traverseTree(x, node->right);
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
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

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);//

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate accuracy using the true labels and predicted labels for the test data
		double test_accuracy = Metrics::accuracy(testLabels, testPredictions);


		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate accuracy using the true labels and predicted labels for the training data
		double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(train_accuracy, test_accuracy,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<double>(),
			std::vector<double>(), std::vector<double>(),
			std::vector<double>());
	}
}