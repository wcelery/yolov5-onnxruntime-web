import React, { useState, useRef } from "react";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/inference";
import "./style/App.css";
import { appConfig } from "./appConfig";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState("Loading OpenCV.js...");
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const imageCanvasRef = useRef(null);
  const textractCanvasRef = useRef(null);

  // configs
  const modelName = appConfig.MODEL_NAME;
  const modelInputShape = appConfig.MODEL_INPUT_SHAPE;

  cv["onRuntimeInitialized"] = async () => {
    // create session
    setLoading("Loading YOLOv5 model...");
    const yolov5 = await InferenceSession.create(
      `${process.env.PUBLIC_URL}/model/${modelName}`
    );

    // warmup model
    setLoading("Warming up model...");
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov5.run({ images: tensor });

    setSession(yolov5);
    setLoading(false);
  };

  return (
    <div className="App">
      {loading && <Loader>{loading}</Loader>}
      <div className="header">
        <h1>Object Detection App</h1>
        <p>
          Object detection application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>
      <div className="content">
        <img
          id="test"
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              imageCanvasRef.current,
              textractCanvasRef.current,
              session,
              modelInputShape
            );
          }}
        />
      </div>
      <canvas id="imageCanvas" ref={imageCanvasRef} />
      <canvas id="textractCanvas" ref={textractCanvasRef} />

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
