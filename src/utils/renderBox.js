import blueprintLabels from "./blueprintLabels.json";

/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvas canvas tag reference
 * @param {Array[Object]} boxes boxes array
 */
export const renderBoxes = async (canvasRef, boxes) => {
  const ctx = canvasRef.getContext("2d");

  boxes.forEach(async (box) => {
    const RED = "#E74C3C";
    const GREEN = "#2ECC71";
    const [x1, y1, width, height] = box.bounding;

    // draw box.
    ctx.fillStyle = hexToRgba(RED, 0.001); // make rect transparent
    ctx.fillRect(x1, y1, width, height);

    // draw border box
    ctx.strokeStyle = RED;
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 400, 2.5);
    ctx.strokeRect(x1, y1, width, height);

    // get response from textract
    const res = await box.textractRequest;
    const foundLabels = res.filter((text) =>
      blueprintLabels.some((label) => text.toLowerCase().includes(label))
    );

    // we can't change the color of the rect, so we have to redraw it on top of old rect
    if (foundLabels.length > 0) {
      ctx.fillStyle = hexToRgba(GREEN, 0.001);
      ctx.fillRect(x1, y1, width, height);
      // draw border box
      ctx.strokeStyle = GREEN;
      ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 400, 2.5);
      ctx.strokeRect(x1, y1, width, height);
    }
  });
};

/**
 * Convert HEX to RGBA
 * @param {string} hex color in #HEX format
 * @param {number} alpha opacity
 */
const hexToRgba = (hex, alpha) => {
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? `rgba(${[
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16),
      ].join(", ")}, ${alpha})`
    : null;
};
