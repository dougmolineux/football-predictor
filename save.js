const fs = require('fs');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs-node');

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

// Prepare data for training
function prepareData(matches, teamToIndex) {
    const inputs = [];
    const outputs = [];

    matches.forEach(match => {
        inputs.push([teamToIndex[match.home_team], teamToIndex[match.away_team]]);
        outputs.push(match.result); // Encoded result
    });

    return {
        inputs: tf.tensor2d(inputs),
        outputs: tf.tensor1d(outputs, 'int32') // Ensure outputs are int32
    };
}

// Create and train the model
async function createModel(inputs, outputs) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [2] }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // 3 classes: win, draw, lose

    model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Convert outputs to float32 for compatibility
    const floatOutputs = outputs.toFloat();

    // Train the model
    await model.fit(inputs, floatOutputs, {
        epochs: 100,
        batchSize: 10,
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

// Main function to execute the program
async function main() {
    const filePath = 'premier-league-matches.csv'; // Your CSV file path
    const modelFilePath = './model'; // Directory to save/load the model

    try {
        const { matches, teams } = await loadData(filePath);
        const teamToIndex = encodeTeams(teams);
        const { inputs, outputs } = prepareData(matches, teamToIndex);

        let model;
        if (fs.existsSync(`${modelFilePath}/model.json`)) {
            // Load the model if it exists
            model = await loadModel(modelFilePath);
            console.log('Model loaded successfully!');
        } else {
            // Train a new model and save it
            model = await createModel(inputs, outputs);
            await saveModel(model, modelFilePath);
            console.log('Model trained and saved successfully!');
        }

        // Example prediction
        const homeTeam = 'Arsenal';
        const awayTeam = 'Manchester Utd';
        const prediction = predictOutcome(model, teamToIndex, homeTeam, awayTeam);

        console.log(`Prediction for ${homeTeam} vs ${awayTeam}:`);
        console.log(`Home Win Probability: ${prediction[0]}`);
        console.log(`Draw Probability: ${prediction[1]}`);
        console.log(`Away Win Probability: ${prediction[2]}`);
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
