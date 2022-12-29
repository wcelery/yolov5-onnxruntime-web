import { DetectDocumentTextCommand, TextractClient } from "@aws-sdk/client-textract";
/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvas canvas tag reference
 * @param {Array[Object]} boxes boxes array
 */
/**
 * Extract text from image with AWS Textract API
 * @param {HTMLImageElement} image image to extract text from
 * @param {HTMLCanvasElement} textractCanvas "buffer" canvas tag reference
 * @param {number} topLeftX Top left X coordinate of the text fragment on the original image
 * @param {number} topLefty Top left Y coordinate of the text fragment on the original image
 * @param {number} bottomRightX Bottom right X coordinate of the text fragment
 * @param {number} bottomRightY Bottom right Y coordinate of the text fragment
 */
export const getTextFromImage = async (
  image,
  textractCanvas,
  topLeftX,
  topLeftY,
  bottomRightX,
  bottomRightY
) => {
  const textractClient = new TextractClient({
    region: "us-east-1",
    credentials: {
      accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY,
      secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY,
    },
  });
  let dst = new cv.Mat();

  // "draw" a rectangle around text
  const rect = new cv.Rect(
    topLeftX,
    topLeftY,
    bottomRightX - topLeftX, // width
    bottomRightY - topLeftY // height
  );

  dst = image.roi(rect); // crop image with rectangle
  cv.imshow("textractCanvas", dst); // load image to "invisible" canvas

  const canvasBlob = await new Promise((resolve) => textractCanvas.toBlob(resolve));

  // free memory
  dst.delete();

  const arr = new Uint8Array(await canvasBlob.arrayBuffer());
  const command = new DetectDocumentTextCommand({
    Document: { Bytes: arr },
  });
  try {
    const data = await textractClient.send(command);
    return data.Blocks.filter((block) => block.BlockType === "LINE").map(
      (block) => block.Text
    );
  } catch (e) {
    console.log(e);
  }
};
