#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox



										///  LinearRegression class implementation  ///


// Function to fit the linear regression model to the training data Matrix Form//
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

    // --- 1. Verificar que los tamaños coincidan
    if (trainData.size() != trainLabels.size()) {
        throw std::invalid_argument("trainData y trainLabels deben tener el mismo número de muestras.");
    }

    int n_samples = trainData.size();
    int n_features = trainData[0].size();

    // --- 2. Construir la matriz de diseño X (con columna de unos para el bias)
    Eigen::MatrixXd X(n_samples, n_features + 1);
    for (int i = 0; i < n_samples; i++) {
        X(i, 0) = 1.0;                          // columna de unos (el bias)
        for (int j = 0; j < n_features; j++) {
            X(i, j + 1) = trainData[i][j];      // las medidas de cada casa
        }
    }

    // --- 3. Convertir las etiquetas (precios) a un vector de Eigen
    Eigen::VectorXd y(n_samples);
    for (int i = 0; i < n_samples; i++) {
        y(i) = trainLabels[i];
    }

    // --- 4. Mínimos cuadrados: coeficientes = (Xᵀ X)⁻¹ Xᵀ y
    m_coefficients = (X.transpose() * X).inverse() * X.transpose() * y;
}


// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

    std::vector<double> result;

    // --- 1. Verificar que el modelo ya fue entrenado
    if (m_coefficients.size() == 0) {
        throw std::runtime_error("El modelo no ha sido entrenado todavía.");
    }

    int n_samples = testData.size();
    int n_features = testData[0].size();

    // --- 2. Construir la matriz de diseño X (igual que en fit: con columna de unos)
    Eigen::MatrixXd X(n_samples, n_features + 1);
    for (int i = 0; i < n_samples; i++) {
        X(i, 0) = 1.0;
        for (int j = 0; j < n_features; j++) {
            X(i, j + 1) = testData[i][j];
        }
    }

    // --- 3. Predicciones = X × coeficientes
    Eigen::VectorXd predictions = X * m_coefficients;

    // --- 4. Convertir el resultado a un std::vector
    for (int i = 0; i < n_samples; i++) {
        result.push_back(predictions(i));
    }

    return result;
}

// Descent Gradient Form
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, double learning_rate, int num_epochs) {

    int n_samples = trainData.size();
    int n_features = trainData[0].size();

    // --- Inicializar coeficientes pequeños (uno por medida + 1 para el bias)
    m_weights.assign(n_features + 1, 0.0);

    // --- NIVEL 1: repetir muchos epochs
    for (int epoch = 0; epoch < num_epochs; epoch++) {

        // --- NIVEL 2: recorrer cada casa
        for (int i = 0; i < n_samples; i++) {

            // Predicción = suma ponderada (SIN sigmoide)
            double prediction = m_weights[0];  // el bias
            for (int j = 0; j < n_features; j++) {
                prediction += m_weights[j + 1] * trainData[i][j];
            }

            // Error = predicción − precio real
            double error = prediction - trainLabels[i];

            // Ajustar coeficientes (coef -= lr × error × medida)
            m_weights[0] -= learning_rate * error;  // el bias
            for (int j = 0; j < n_features; j++) {
                m_weights[j + 1] -= learning_rate * error * trainData[i][j];
            }
        }
    }
}

// Predict function for gradient descent form
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData, bool useGradientDescent) {

    std::vector<double> result;
    int n_features = testData[0].size();

    // Por cada casa, calcular la suma ponderada con los coeficientes aprendidos
    for (int i = 0; i < (int)testData.size(); i++) {
        double prediction = m_weights[0];  // el bias
        for (int j = 0; j < n_features; j++) {
            prediction += m_weights[j + 1] * testData[i][j];
        }
        result.push_back(prediction);
    }

    return result;
}


/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
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