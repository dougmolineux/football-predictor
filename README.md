# Football Predictor

![Premier League Logo](https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg)

This project uses TensorFlow.js to create a machine learning model that predicts the outcomes of Premier League matches based on historical data.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Example](#example)
- [Results](#results)
- [Screenshots](#screenshots)
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

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
