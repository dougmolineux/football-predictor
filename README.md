# Football Predictor

![Premier League Logo](https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg)

This project uses TensorFlow.js to create a machine learning model that predicts the outcomes of Premier League matches based on historical data.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
- [Results](#results)
- [Screenshots](#screenshots)
- [Notes on Tensorflow Implementation](#notes-on-tensorflow-implementation)
  - [Adding the Layers](#adding-the-layers)
  - [Configuring the Model](#configuring-the-model)
  - [Summary of Implementation](#summary-of-implementation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The Premier League Match Predictor uses historical match data to train a neural network model that can predict the result of a match between two given teams. The model predicts the likelihood of a home win, draw, or away win.

## Setup

### Prerequisites
- Node.js
- NPM

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/dougmolineux/football-predictor.git
    cd football-predictor
    ```
2. Install the required dependencies:
    ```bash
    npm install
    ```

3. Ensure you have a CSV file named `premier-league-matches.csv` in the project directory with the following format:
    ```csv
    Season_End_Year,Wk,Date,Home,HomeGoals,AwayGoals,Away,FTR
    1993,1,1992-08-15,Coventry City,2,1,Middlesbrough,H
    1993,1,1992-08-15,Leeds United,2,1,Wimbledon,H
    1993,1,1992-08-15,Sheffield Utd,2,1,Manchester Utd,H
    1993,1,1992-08-15,Crystal Palace,3,3,Blackburn,D
    1993,1,1992-08-15,Arsenal,2,4,Norwich City,A
    ```

## Usage

### Training the Model
Run the following command to train the model:
```bash
node main.js
```

### Results
The model will output the probabilities of a home win, draw, and away win based on the input teams' historical performance.

## Screenshots
<img src='https://github.com/dougmolineux/football-predictor/blob/015d5c7bc1d84608f8f6a4e3ab23f107e122b227/screenshots/example.png' />

## Notes on Tensorflow Implementation
### Adding the Layers
```
const model = tf.sequential();
```
`tf.sequential()`: This creates a sequential model, which is a linear stack of layers. A sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
```
model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [2] }));
model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
```
`model.add(tf.layers.dense({ ... }))`: This adds a dense (fully connected) layer to the model.
```
tf.layers.dense({
    units: 10,          // specifies that the layer has 10 neurons.
    activation: 'relu', // Rectified Linear Unit
    inputShape: [2]     // input sample will have 2 features (home team index and away team index)
})
```
`units: 10`: This specifies that the layer has 10 neurons. Why 10? With 10 neurons, the model has enough capacity to capture these relationships without being too complex. Starting with 10 neurons can be a reasonable initial choice, which can be adjusted based on the model's performance during validation. By having 10 neurons, the layer can introduce sufficient non-linearity to the model. This is particularly important for problems where the relationship between input features and the output is not linear.

`activation: 'relu'`: This specifies the activation function for the layer. 'relu' stands for Rectified Linear Unit, which is commonly used in hidden layers of neural networks.

`inputShape: [2]`: This specifies that the input to this layer has a shape of [2]. In this case, it means that each input sample will have 2 features (home team index and away team index).

```
tf.layers.dense({ units: 3, activation: 'softmax' })
```
`units: 3`: This specifies that the layer has 3 neurons, which correspond to the 3 possible classes (home win, draw, away win).

`activation: 'softmax'`: This specifies the activation function for the layer. 'softmax' is used in the output layer for classification problems because it converts the outputs to a probability distribution over the 3 classes.

### Configuring the Model
```
model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
});
```
`model.compile({ ... })`: This configures the model for training.

`optimizer: 'adam'`: This specifies the optimizer to use during training. 'adam' is an adaptive learning rate optimization algorithm that's popular for training deep learning models. The Adam optimizer, or Adaptive Moment Estimation, is a machine learning algorithm used to train deep neural networks. 

`loss: 'sparseCategoricalCrossentropy'`: This specifies the loss function to use during training. 'sparseCategoricalCrossentropy' is appropriate for classification problems where the target variable is an integer representing the class index.

`metrics: ['accuracy']`: This specifies the metrics to evaluate during training and testing. 'accuracy' measures the fraction of correctly predicted instances.

### Summary of Implementation
1. Define Model: `const model = tf.sequential();` creates a new sequential model.
2. Add Layers:
   
    a. First layer with 10 neurons, ReLU activation, and input shape of 2.
   
    b. Second layer with 3 neurons and softmax activation for classification.
   
4. Compile Model: Configures the model with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric.

These steps set up the neural network for training on the football match data to predict match outcomes.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
