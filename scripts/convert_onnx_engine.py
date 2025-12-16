from ultralytics import YOLO
import os

def convert_onnx():
    model_path = "/root/yolo_onnx_models/yolo11m.pt"
    model = YOLO(model_path)
    # 导出模型为 ONNX 格式
    # model.export(format="onnx")
    # 导出为tensorRT格式
    model.export(format="engine",
                 dynamic=True,
                 batch=64,
                 #  workspace=4,
                 #  half=False,
                 #  int8=False,
                 nms=True)
    engine_path = os.path.join(os.path.dirname(
        model_path), os.path.basename(model_path).replace(".pt", ".engine"))
    print(engine_path)


def yolo_demo():
    model_path = "/root/yolo_onnx_models/yolo11m.engine"
    image_path = "/root/My_project/DemoTensorRT/images/bus.jpg"
    model = YOLO(model_path)
    results = model(image_path)
    print(results[0].boxes)

def main():
    # convert_onnx()
    yolo_demo()



if __name__ == "__main__":
    main()
