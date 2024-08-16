const fs = require('fs');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs-node');

const config = {
    neuronCount: 128,  // Updated in version 3: Increased from 64 to 128 for better capacity
    batchSize: 16,     // Updated in version 3: Reduced batch size for more frequent updates
    epochs: 50,        // Updated in version 3: Increased to 50 for more training without overfitting
    learningRate: 0.0005,  // Updated in version 3: Lower learning rate for finer adjustments
    dropoutRate: 0.3,  // Updated in version 3: Increased dropout to prevent overfitting
};

// Load and preprocess data
async function loadData(filePath) {
    const matches = [];
    const teams = new Set();

    return new Promise((resolve, reject) => {
        fs.createReadStream(filePath)
            .pipe(csv())
            .on('data', (row) => {
                // Collect unique team names
                teams.add(row.Home);
                teams.add(row.Away);

                // Convert columns to numerical data
                matches.push({
                    season_end_year: parseInt(row.Season_End_Year, 10),
                    week: parseInt(row.Wk, 10),
                    date: row.Date,
                    home_team: row.Home,
                    away_team: row.Away,
                    home_goals: parseInt(row.HomeGoals, 10),
                    away_goals: parseInt(row.AwayGoals, 10),
                    result: row.FTR === 'H' ? 1 : (row.FTR === 'A' ? -1 : 0)
                });
            })
            .on('end', () => {
                resolve({ matches, teams: Array.from(teams) });
            })
            .on('error', (error) => {
                reject(error);
            });
    });
}

// Encode team names into numerical values and normalize
function encodeTeams(teams) {
    const teamToIndex = {};
    teams.forEach((team, index) => {
        teamToIndex[team] = index / teams.length;  // Normalize team index
    });
    return teamToIndex;
}

// Prepare data for training and testing
function prepareData(matches, teamToIndex) {
    const inputs = [];
    const outputs = [];
    const testInputs = [];
    const testOutputs = [];

    matches.forEach(match => {
        const input = [teamToIndex[match.home_team], teamToIndex[match.away_team]];
        const output = match.result;

        if (match.season_end_year < 2023) {
            inputs.push(input);
            outputs.push(output);
        } else if (match.season_end_year === 2023) {
            testInputs.push(input);
            testOutputs.push(output);
        }
    });

    return {
        trainingData: {
            inputs: tf.tensor2d(inputs),
            outputs: tf.tensor1d(outputs, 'int32')
        },
        testData: {
            inputs: tf.tensor2d(testInputs),
            outputs: testOutputs // Keep as an array for easy comparison later
        }
    };
}

// Create and train the model
async function createModel(inputs, outputs) {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: config.neuronCount,
        activation: 'relu',
        inputShape: [2]
    }));

    model.add(tf.layers.dense({
        units: config.neuronCount,
        activation: 'relu'
    }));

    model.add(tf.layers.dropout({ rate: config.dropoutRate }));  // Updated in version 3: Increased dropout rate

    model.add(tf.layers.dense({
        units: config.neuronCount,
        activation: 'relu'
    })); // Updated in version 3: Added another hidden layer

    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    })); // 3 units: win, draw, lose

    const optimizer = tf.train.adam(config.learningRate);  // Updated in version 3: Adjusted learning rate
    model.compile({
        optimizer: optimizer,
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Convert outputs to float32 for compatibility
    const floatOutputs = outputs.toFloat();

    const earlyStopping = tf.callbacks.earlyStopping({
        monitor: 'loss',
        patience: 10,
    });

    // Train the model
    await model.fit(inputs, floatOutputs, {
        epochs: config.epochs,
        batchSize: config.batchSize,
        shuffle: true,
        callbacks: [earlyStopping]
    });

    return model;
}

// Save the model to the local file system
async function saveModel(model, filePath) {
    await model.save(`file://${filePath}`);
}

// Load the model from the local file system
async function loadModel(filePath) {
    return await tf.loadLayersModel(`file://${filePath}/model.json`);
}

// Function to predict the outcome for a given home and away team
function predictOutcome(model, teamToIndex, homeTeam, awayTeam) {
    const homeIndex = teamToIndex[homeTeam];
    const awayIndex = teamToIndex[awayTeam];
    const inputTensor = tf.tensor2d([[homeIndex, awayIndex]]);
    const prediction = model.predict(inputTensor);
    return prediction.arraySync()[0]; // Get prediction as an array
}

// Calculate the accuracy of the model's predictions
function calculateAccuracy(predictions, actualResults) {
    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
        const predictedResult = predictions[i].indexOf(Math.max(...predictions[i])) - 1; // -1 to adjust the index to [-1, 0, 1]
        if (predictedResult === actualResults[i]) {
            correct++;
        }
    }
    return (correct / predictions.length) * 100;
}

// Main function to execute the program
async function main() {
    const filePath = 'premier-league-matches.csv'; // Your CSV file path
    const modelFilePath = './model'; // Directory to save/load the model

    try {
        const { matches, teams } = await loadData(filePath);
        const teamToIndex = encodeTeams(teams);
        const { trainingData, testData } = prepareData(matches, teamToIndex);

        let model;
        if (fs.existsSync(`${modelFilePath}/model.json`)) {
            // Load the model if it exists
            model = await loadModel(modelFilePath);
            console.log('Model loaded successfully!');
        } else {
            // Train a new model and save it
            model = await createModel(trainingData.inputs, trainingData.outputs);
            await saveModel(model, modelFilePath);
            console.log('Model trained and saved successfully!');
        }

        // Predict outcomes for the test data
        const predictions = model.predict(testData.inputs).arraySync();

        // Calculate accuracy
        const accuracy = calculateAccuracy(predictions, testData.outputs);
        console.log(`Model accuracy for 2023 matches: ${accuracy.toFixed(2)}%`);

    } catch (error) {
        console.error('Error:', error);
    }
}

main();
