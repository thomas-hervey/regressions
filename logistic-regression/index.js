require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const plot = require('node-remote-plot')

let { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions: value => {
        return value === 'TRUE' ? 1 : 0
      }
    }
  })

const logisticRegression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50
})

logisticRegression.train()
const accuracy = logisticRegression.test(testFeatures, testLabels)
console.log('Accuracy is', accuracy)

plot({
  x: logisticRegression.costHistory.reverse()
})




