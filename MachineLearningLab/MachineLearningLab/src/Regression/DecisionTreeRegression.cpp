#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
using namespace System::Windows::Forms; // For MessageBox



///  DecisionTreeRegression class implementation  ///


// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{
}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {

	std::vector<double> predictions;
	
	for (int i = 0; i < (int)X.size(); i++) {
		double pred = traverseTree(X[i], root);
		predictions.push_back(pred);
	}
	return predictions;
}


// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {

	int n_samples = X.size();
	int split_idx = -1;
	double split_thresh = 0.0;

	// --- criterios de parada → HOJA con el promedio
	if (depth >= max_depth || n_samples < min_samples_split) {
		double leaf_value = mean(y);
		return new Node(0, 0.0, nullptr, nullptr, leaf_value);
	}

	// --- buscar el split con MENOR error
	double best_mse = 1e18; // el peor caso posible
	for (int feat = 0; feat < n_feats; feat++) {
		std::vector<double> X_column;
		for (int i = 0; i < n_samples; i++) {
			X_column.push_back(X[i][feat]);
		}
		for (double threshold : X_column) {
			double mse = meanSquaredError(y, X_column, threshold);
			if (mse < best_mse) {          // ← ahora buscamos el MENOR
				best_mse = mse;
				split_idx = feat;
				split_thresh = threshold;
			}
		}
	}

	// --- si no se encontró un split útil → HOJA
	if (split_idx == -1) {
		double leaf_value = mean(y);
		return new Node(0, 0.0, nullptr, nullptr, leaf_value);
	}

	// --- partir los datos según el mejor split
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
	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {

	double mse = 0.0;
	// separar los valores en izquierda y derecha según el umbral
	std::vector<double> left, right;
	for (int i = 0; i < (int)X_column.size(); i++) {
		if (X_column[i] <= split_thresh)
			left.push_back(y[i]);
		else
			right.push_back(y[i]);
	}

	// si un lado queda vacío, el split no sirve → error gigante para descartarlo
	if (left.empty() || right.empty())
		return 1e18;

	// dispersión de cada lado: promedio de (valor − promedio)²
	double mean_left = mean(left);
	double mean_right = mean(right);
	double mse_left = 0.0, mse_right = 0.0;
	for (double v : left)  mse_left += (v - mean_left) * (v - mean_left);
	for (double v : right) mse_right += (v - mean_right) * (v - mean_right);
	mse_left /= left.size();
	mse_right /= right.size();

	// error combinado, ponderado por tamaño
	int n = y.size();
	mse = ((double)left.size() / n) * mse_left + ((double)right.size() / n) * mse_right;

	
	return mse;
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double meanValue = 0.0;
	if (values.empty()) return 0.0;

	double sum = 0.0;
	for (double v : values) {
		sum += v;                     // sumar todos
	}
	meanValue = sum / values.size();  // dividir entre cuántos hay
	
	return meanValue;
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {

	if (node->isLeafNode()) {
		return node->value;
	}
	if (x[node->feature] <= node->threshold) {
		return traverseTree(x, node->left);
	}
	return traverseTree(x, node->right);
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
	std::vector<double>, std::vector<double>,
	std::vector<double>, std::vector<double>>
	DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
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
		// Load the dataset from the file path
		std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

		// Convert the dataset from strings to doubles
		std::vector<std::vector<double>> dataset;
		bool isFirstRow = true; // Flag to identify the first row

		for (const auto& row : data) {
			if (isFirstRow) {
				isFirstRow = false;
				continue; // Skip the first row (header)
			}

			std::vector<double> convertedRow;
			for (const auto& cell : row) {
				try {
					double value = std::stod(cell);
					convertedRow.push_back(value);
				}
				catch (const std::exception& e) {
					// Handle the exception or set a default value
					std::cerr << "Error converting value: " << cell << std::endl;
					// You can choose to set a default value or handle the error as needed
				}
			}
			dataset.push_back(convertedRow);
		}

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate evaluation metrics (e.g., MAE, MSE)
		double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
		double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
		double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate evaluation metrics for training data
		double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
		double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
		double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(test_mae, test_rmse, test_rsquared,
			train_mae, train_rmse, train_rsquared,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			std::vector<double>(), std::vector<double>(),
			std::vector<double>(), std::vector<double>());
	}
}

