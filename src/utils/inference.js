import { Tensor } from "onnxruntime-web";
import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox";
import { getTextFromImage } from "./textract";
import { appConfig } from "../appConfig";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} imageCanvas canvas render image and boxes over it
 * @param {HTMLCanvasElement} textractCanvas buffer-like canvas to extract textboxes from
 * @param {ort.InferenceSession} session YOLOv5 onnxruntime session
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  imageCanvas,
  textractCanvas,
  session,
  inputShape
) => {
  const mat = cv.imread(image);
  const matBig = new cv.Mat();
  const matSmall = new cv.Mat();

  const bigSize = new cv.Size(
    mat.cols * appConfig.BIG_IMAGE_SIZE,
    mat.rows * appConfig.BIG_IMAGE_SIZE
  );
  const coef = mat.cols / appConfig.SMALL_IMAGE_SIZE;
  const smallSize = new cv.Size(appConfig.SMALL_IMAGE_SIZE, appConfig.SMALL_IMAGE_SIZE); // 640 x 640
  const smallSize1 = new cv.Size(appConfig.SMALL_IMAGE_SIZE, mat.rows / coef); // 640 x (height of the original image / coef)

  cv.resize(mat, matBig, bigSize, 0, 0, cv.INTER_AREA);
  cv.cvtColor(matBig, matBig, cv.COLOR_BGR2RGB);
  const h0 = matBig.rows;
  const w0 = matBig.cols;

  cv.resize(matBig, matSmall, smallSize1, 0, 0, cv.INTER_AREA); // resize big image to 640 x 426
  cv.resize(matSmall, matSmall, smallSize, 0, 0, cv.INTER_AREA); // resize small (640 x 426) image to 640 x 640

  const h = matSmall.rows;
  const w = matSmall.cols;

  cv.imshow("imageCanvas", matBig);

  const input = cv.blobFromImage(
    matSmall,
    1 / 255, // normalize 0..255 -> 0..1
    new cv.Size(w, h),
    new cv.Scalar(0, 0, 0),
    false, // swaps bgr to rgb
    false // crops the image (or matrix?)
  );

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const { output0 } = await session.run({ images: tensor });

  const coordinatesArr = [];
  const coordinatesWithDimsArr = [];
  const confidencesArr = [];

  for (let i = 0; i < output0.size /* = 64512 */; i += output0.dims[2] /* = 6 */) {
    const [x, y, w, h, confidence, classId] = output0.data.slice(i, i + output0.dims[2]); // [0 - 6]
    const x1 = x - w / 2;
    const y1 = y - h / 2;
    const x2 = x + w / 2;
    const y2 = y + h / 2;

    coordinatesArr.push([x1, y1, x2, y2]);
    coordinatesWithDimsArr.push([x1, y1, x2, y2, w, h]);
    confidencesArr.push(confidence);
  }

  const boxes = [];

  const indexes = await tf.image.nonMaxSuppressionAsync(
    tf.tensor2d(coordinatesArr), // make 2d tensor from regular 2d JS array
    tf.tensor1d(confidencesArr), // make 1d tensor from regular 1d JS array
    appConfig.NMS_MAX_BOXES,
    appConfig.IOU_THRES,
    appConfig.MIN_CONFIDENCE
  );
  const foundTables = await indexes.array(); // return indexes of the table in original tensor data

  foundTables.forEach((table, idx) => {
    const [x1, y1, x2, y2, w, h] = coordinatesWithDimsArr[table]; // coordinates for small image size

    // scale coordinates up for big image
    const topLeftX = parseInt((x1 / appConfig.SMALL_IMAGE_SIZE) * w0);
    const topLeftY = parseInt((y1 / appConfig.SMALL_IMAGE_SIZE) * h0);
    const bottomRightX = parseInt((x2 / appConfig.SMALL_IMAGE_SIZE) * w0);
    const bottomRightY = parseInt((y2 / appConfig.SMALL_IMAGE_SIZE) * h0);

    // calculate width and height for context.strokeRect
    const rectWidth = parseInt((w / appConfig.SMALL_IMAGE_SIZE) * w0);
    const rectHeight = parseInt((h / appConfig.SMALL_IMAGE_SIZE) * h0);

    // calculate coordinates for table title box
    let x0_ =
      parseInt(topLeftX - appConfig.X_FRAME * w0) < 0
        ? 0
        : parseInt(topLeftX - appConfig.X_FRAME * w0);
    let y0_ =
      parseInt(topLeftY - appConfig.TOP_FRAME * h0) < 0
        ? 0
        : parseInt(topLeftY - appConfig.TOP_FRAME * h0);
    let x1_ =
      parseInt(bottomRightX + appConfig.X_FRAME * w0) > w0
        ? w0
        : parseInt(bottomRightX + appConfig.X_FRAME * w0);
    let y1_ =
      parseInt(topLeftY + appConfig.BOTTOM_FRAME * h0) > h0
        ? h0
        : parseInt(topLeftY + appConfig.BOTTOM_FRAME * h0);

    boxes.push({
      bounding: [topLeftX, topLeftY, rectWidth, rectHeight],
      textractRequest: getTextFromImage(matBig, textractCanvas, x0_, y0_, x1_, y1_), // do not wait for promise, draw rectangles right away
    });
  });

  mat.delete();
  matBig.delete();
  matSmall.delete();
  renderBoxes(imageCanvas, boxes); // Draw boxes
};
