// handpose
let handpose;
let video;
let predictions = [];

// knn
// Create a KNN classifier
const knnClassifier = ml5.KNNClassifier();
let loopBroken = false;

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  handpose = ml5.handpose(video, modelReady);
  // This sets up an event that fills the global variable "predictions"
  // with an array every time new hand poses are detected
  handpose.on('predict', results => {
    predictions = results;
  });

  // Hide the video element, and just show the canvas
  video.hide();
  // p5 how to set up button
  createUI();
}

function draw() {
  image(video, 0, 0, width, height);
  keyPresses();
  drawKeypoints();
  restartClassifier();
}

function restartClassifier() {
  if (loopBroken) {
    loopBroken = false;
    classify();
  }
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];

    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      fill(0, 255, 0);
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
  }
}

// 1. train
function keyPresses() {
  // A is pressed
  if (keyIsDown(65)) {
    train('A');
  }
  // B is pressed
  if (keyIsDown(66)) {
    train('B');
  }
  // C is pressed
  if (keyIsDown(67)) {
    train('C');
  }
}

function train(label) {
  // check if we have data returned from handpose
  if (predictions[0] != undefined) {
    const prediction = predictions[0].landmarks;
    knnClassifier.addExample(prediction, label);
    console.log('trained an example for pose ' + label);
  } else {
    console.log('no data supplied for training');
  }
}

// 2. classify
// Predict the current frame.
function classify() {
  // Get the total number of labels from knnClassifier
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error('There are no examples in any label');
    return;
  }
  // check if we have data returned from handpose
  console.log(predictions[0]);
  if (predictions[0] != undefined) {
    const prediction = predictions[0].landmarks;
    // Use knnClassifier to classify which label do these features belong to
    // You can pass in a callback function `gotResults` to knnClassifier.classify function
    knnClassifier.classify(prediction, gotResults);
  } else {
    loopBroken = true;
  }
}

// Show the results
function gotResults(err, result) {
  // Display any error
  if (err) {
    console.error(err);
  } else if (result.confidencesByLabel) {
    const confidences = result.confidencesByLabel;
    console.log(confidences);
  }
  classify();
}

function createUI() {
  createButton('Predict')
    .mousePressed(classify)
    .id('predict-button')
    .parent('ui');
  createP('Press a, b or c to train examples').id('training-info').parent('ui');
}

function modelReady() {
  console.log('Model ready!');
}
