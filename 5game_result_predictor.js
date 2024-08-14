const fs = require('fs');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs-node');
const { log } = require('console');

const config = {
    neuronCount: 10,
    batchSize: 10,
    epochs: 1000,
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

// Encode team names into numerical values
function encodeTeams(teams) {
    const teamToIndex = {};
    teams.forEach((team, index) => {
        teamToIndex[team] = index;
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
        }
        // } else if (match.season_end_year === 2023) {
        //     testInputs.push(input);
        //     testOutputs.push(output);
        // }
    });

    return {
        trainingData: {
            inputs: tf.tensor2d(inputs),
            outputs: tf.tensor1d(outputs, 'int32')
        }
        // testData: {
        //     inputs: tf.tensor2d(testInputs),
        //     outputs: testOutputs // Keep as an array for easy comparison later
        // }
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
    })); // Add another hidden layer

    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    })); // 3 units: win, draw, lose

    model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Convert outputs to float32 for compatibility
    const floatOutputs = outputs.toFloat();

    // Train the model
    await model.fit(inputs, floatOutputs, {
        epochs: config.epochs,
        batchSize: config.batchSize,
        shuffle: true
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

        console.log('Predictions for the First week of the 2024 / 2025 Season')
        predictWrapper('Manchester Utd', 'Fulham', model, teamToIndex);
        predictWrapper('Ipswich Town', 'Liverpool', model, teamToIndex);
        predictWrapper('Arsenal', 'Wolves', model, teamToIndex);
        predictWrapper('Everton', 'Brighton', model, teamToIndex);
        predictWrapper('Newcastle Utd', 'Southampton', model, teamToIndex);
        predictWrapper('Nott\'ham Forest', 'Bournemouth', model, teamToIndex);
        predictWrapper('West Ham', 'Aston Villa', model, teamToIndex);
        predictWrapper('Brentford', 'Crystal Palace', model, teamToIndex);
        predictWrapper('Chelsea', 'Manchester City', model, teamToIndex);
        predictWrapper('Leicester City', 'Tottenham', model, teamToIndex);
    } catch (error) {
        console.error('Error:', error);
    }
}

function predictWrapper(homeTeam, awayTeam, model, teamToIndex) {
    const prediction = predictOutcome(model, teamToIndex, homeTeam, awayTeam);

    console.log(`Prediction for ${homeTeam} vs ${awayTeam}:`);
    console.log(` Home Win Probability: ${prediction[0]}`);
    console.log(` Draw Probability: ${prediction[1]}`);
    console.log(` Away Win Probability: ${prediction[2]}`);
}

main();
