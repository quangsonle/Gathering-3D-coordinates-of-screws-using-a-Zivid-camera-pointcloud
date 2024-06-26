This is a web-server-based application that utilizes YOLOv5 to detect screws and holes, returning their 3D coordinates (x, y, z) within the frame and saving them in a JSON file.

The Zivid camera provides an RGB resolution of 2048x2448, but for point-cloud frames in "Subsample mode", it offers 1224x1024. To circumvent the need for a homography transformation from the RGB frame to the point-cloud frame for gathering corresponding point-cloud data, the RGB frames are generated from the Red, Green, and Blue components of the point-cloud PLY files, ensuring homogeneity.

The Z-value of the coordinates is a critical factor that can serve as a filter to enhance detection: for a true screw, the Z variance within the bounding box must be significant. Additionally, the Z-value can confirm whether a screw is within the operational area by ensuring it falls within a specific range.

Once all screws in the operational area are confirmed with their 3D coordinates, a robotic arm can be programmed to move and unscrew them automatically.
1. Running the Application

Note: Ensure that Python and pip are installed on your system, for both Windows and Linux.

- Install YOLOv5 and its dependencies:

```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -qr requirements.txt
```

- Install Flask and dotenv:

```
pip install flask
pip install python-dotenv
```

- Install PLY:

```
pip3 install plyfile
```

- Set up the server:

Download the model (best.pt) from the provided Google Drive link and place it in the server folder.

https://drive.google.com/file/d/1cJRXeKsomoV8qAv_4y0yFeDFbli-bbt2/view?usp=sharing

Navigate to the server directory and run "flask run".

If the message " * Running on http://127.0.0.1:8000 (Press CTRL+C to quit)" is displayed, the server is operational.

Access the server by navigating to "127.0.0.1:8000" in a web browser.

2. Codebase Overview
- infer.py contains the declarations for the YOLO class and methods.
- server.py is the backend of the server.
- The templates folder contains index.html, which is the frontend interface of the server.
The showcase, which ran on a CPU, could see performance improvements if it operated on a GPU using more optimal model formats, such as ONNX or TensorRT, instead of Torch.

Two samples (pc2.ply and pc4.ply) are provided for a validation 

Video: https://www.youtube.com/watch?v=Oi7rD8fVt1s

These instructions do not cover how to annotate and train the data to create the model.
