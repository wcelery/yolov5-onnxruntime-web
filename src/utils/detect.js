import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv7 onnxruntime session
 * @param {Number} classThreshold class threshold
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (image, canvas, session, classThreshold, inputShape) => {
  const [modelWidth, modelHeight] = inputShape.slice(2);

  const mat = cv.imread(image); // read from img tag
  const matC3 = new cv.Mat(modelWidth, modelHeight, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR
  const input = cv.blobFromImage(
    matC3,
    1 / 255.0,
    new cv.Size(modelWidth, modelHeight),
    new cv.Scalar(0, 0, 0),
    true,
    false
  ); // preprocessing image matrix

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const { output } = await session.run({ images: tensor }); // run session and get output layer

  const boxes = [];

  // looping through output
  for (let r = 0; r < output.size; r += output.dims[1]) {
    const data = output.data.slice(r, r + output.dims[1]); // get rows
    const [x0, y0, x1, y1, classId, score] = data.slice(1);
    const w = x1 - x0,
      h = y1 - y0;

    if (score >= classThreshold)
      boxes.push({
        classId: classId,
        probability: score,
        bounding: [x0, y0, w, h],
      });
  }

  renderBoxes(canvas, boxes); // Draw boxes
};
