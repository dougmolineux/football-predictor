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
  - [Instantiating a Sequential Model](#instantiating-a-sequential-model)
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

## Results
The model will output the probabilities of a home win, draw, and away win based on the input teams' historical performance.

To check the accuracy of the model, a script called `comparison.js` uses all the results from before 2023, then predicts the 2023 results. The last step compares the predictions with the real results and prints an accuracy percentage, where 100% would mean it predicted all results accurately. The first version of this model (which is what is configured in `main.js`) resulted in a 22% accuracy rating (which can be seen in the screenshot below).

This performance is actually quite terrible, and even worse than a random number generator, which would have a 33 percent accuracy (considering there are 3 outcomes: home win, away win, draw).

### Questions: 
1. Does the accuracy of the model change between training sessions, even if the configuration of the model, the dataset and everything stays the same?

Answer: The first time we trained the model we got a 22.89% accuracy, the second time we ran it we got a 22.63% accuracy. It appears to be roughly the same, but slightly different.

2. Does the accuracy improve if we start to remove the earlier seasons (newer performances are more relevant)?

Answer: Removing all the games between 1993 and 1999 improves the accuracy by 1 percent (23.42%).

3. How long does it take to train with the current configuration?

Answer: It takes about 2.5 seconds per epoch, with 100 epochs to train it in its current state. 250 seconds, a bit more than 4 minutes

4. How much accuracy does 200 epochs give us (100 give us roughly 22%)

Answer: 200 epochs produced an accuracy of 23.95%, whereas previously it was 23.42%. 200 epochs gave a slight increase in accuracy of half a percent.

5. There's a theory that batch size can improve accuracy, but increase training time. How much more accurate does batch size improve accuracy?

Answer: batch size of 20 actually decreased it from 23 to 22#. 40 batch size decreases training time down to 500ms per epoch. So the higher the batch size, the faster the training is, it also doesn't seem to affect accuracy significantly. A batch size of 4 actually takes 5.5 seconds per epoch, but accuracy remained around 22 percent.

6. What impact will increasing the number of neurons have on this process?

Answer: Doubling the neurons to 20 (keeping batch size at 40), actually got 23.95% accuracy, which is the highest of seen so far. 40 neurons had exactly the same. Just for fun i tried 100 neurons, but the accuracy actually went down.

7. What about adding another layer in between the input layer and output layer?

Answer: Adding a layer with the same neuron count had no impact on accuracy, but did increase training time.

8. What if you only use results after 2010?

### Screenshots
<img src='https://github.com/dougmolineux/football-predictor/blob/015d5c7bc1d84608f8f6a4e3ab23f107e122b227/screenshots/example.png' />
<img src='https://github.com/dougmolineux/football-predictor/blob/97d8a5067896ae6fb2234e0ade8f4571f4fe7d65/screenshots/accuracy_of_first_model.png' />

## Notes on Tensorflow Implementation
### Instantiating a Sequential Model
```
const model = tf.sequential();
```
`tf.sequential()`: This creates a sequential model, which is a linear stack of layers. A sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
### Adding the Layers
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
