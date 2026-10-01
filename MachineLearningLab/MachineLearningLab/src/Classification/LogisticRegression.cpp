#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map>
#include <cstdlib>

using namespace System::Windows::Forms; // For MessageBox


                                            ///  LogisticRegression class implementation  ///
// Constractor

LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {}

// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    int num_features = X_train[0].size();
    int num_classes = std::set<double>(y_train.begin(), y_train.end()).size();

    // --- Inicializar pesos ALEATORIOS para cada clase (+1 por el bias)
    std::srand(42); // semilla fija → resultados repetibles (útil para comparar en la Task 2)
    weights.clear();
    for (int c = 0; c < num_classes; c++) {
        std::vector<double> class_weights;
        for (int j = 0; j < num_features + 1; j++) {
            double r = ((double)std::rand() / RAND_MAX) * 0.02 - 0.01; // valor pequeño entre -0.01 y 0.01
            class_weights.push_back(r);
        }
        weights.push_back(class_weights);
    }

    // --- NIVEL 1: un clasificador por cada clase (one-vs-rest)
    for (int c = 0; c < num_classes; c++) {

        // --- NIVEL 2: repetir muchos epochs
        for (int epoch = 0; epoch < num_epochs; epoch++) {

            // --- NIVEL 3: recorrer cada flor de entrenamiento
            for (int i = 0; i < (int)X_train.size(); i++) {

                // Convertir a problema binario: 1 si ES la clase c, 0 si no
				double y_binary = ((int)y_train[i] == c) ? 1.0 : 0.0;  //valor real de la etiqueta para esta clase

                // Suma ponderada z (empieza con el bias = peso [0])
                double z = weights[c][0];
                for (int j = 0; j < num_features; j++) {
                    z += weights[c][j + 1] * X_train[i][j];
                }

                // Predicción = sigmoide(z)
                double prediction = sigmoid(z);

                // Error = predicción − real
                double error = prediction - y_binary;

                // Ajustar pesos con gradient descent (peso -= lr × error × medida)
                weights[c][0] -= learning_rate * error;           // el bias (su "medida" es 1)
                for (int j = 0; j < num_features; j++) {
                    weights[c][j + 1] -= learning_rate * error * X_train[i][j];
                }
            }
        }
    }
}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;

    int num_features = X_test[0].size();
    int num_classes = weights.size();

    // Recorrer cada flor de test
    for (int i = 0; i < (int)X_test.size(); i++) {

        double best_score = -1e18;  // el peor puntaje posible
        int best_class = 0;

        // Calcular el puntaje de cada clase y quedarse con el mayor
        for (int c = 0; c < num_classes; c++) {
            double z = weights[c][0];  // bias
            for (int j = 0; j < num_features; j++) {
                z += weights[c][j + 1] * X_test[i][j];
            }
            if (z > best_score) {   // ¿este modelo está más seguro?
                best_score = z;
                best_class = c;
            }
        }

        predictions.push_back((double)best_class);  // la clase ganadora
    }

    return predictions;
}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

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
        fit(trainData, trainLabels);

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