import * as tf from '@tensorflow/tfjs-node';

const temperatures = [20, 21, 22, 23];
const salesRate = [40, 42, 44, 46];

const src = tf.tensor(temperatures);
const dst = tf.tensor(salesRate);

const X = tf.input({ shape: [1] });
const Y = tf.layers.dense({ units: 1 }).apply(X);
const model = tf.model({ inputs: X, outputs: Y as tf.SymbolicTensor });
const compileParam = {
  optimizer: tf.train.adam(),
  loss: tf.losses.meanSquaredError,
};
model.compile(compileParam);

function runFit(times?: number) {
  const fitParam = {
    epochs: times ? times * 100 : 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log('epoch', epoch, logs, 'RMSE: ', Math.sqrt(logs.loss));
      },
    },
  };
  model.fit(src, dst, fitParam).then(result => {
    const predictData = model.predict(src);
    (predictData as tf.Tensor<tf.Rank>).print();
  });
}

function newTest() {
  const newData = [30, 60, 72, 127];
  const transData = tf.tensor(newData);
  const predictData = model.predict(transData);
  (predictData as tf.Tensor<tf.Rank>).print();
}
