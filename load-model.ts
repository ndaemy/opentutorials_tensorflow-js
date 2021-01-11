import * as tf from '@tensorflow/tfjs-node';

async function predict(input: number) {
  const model = await tf.loadLayersModel(
    'file:///Users/fly/Documents/dev/opentutorials_tensorflow-js/model.json',
  );
  const src = tf.tensor([input]);
  const predictData = model.predict(src);
  (predictData as tf.Tensor<tf.Rank>).print();
}
