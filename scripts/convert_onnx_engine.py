from ultralytics import YOLO
import os


def engine_post_process(engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
        # 去除头部的json字段内容
        json_end_index = engine_data.find(b"}ftrt") + 1
        engine_data = engine_data[json_end_index:]

    # 保存处理后的引擎文件
    with open(engine_path, "wb") as f:
        f.write(engine_data)


def convert_tensorRT_engine():
    model_path = "/root/yolo_onnx_models/yolo11n.pt"
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
    engine_post_process(engine_path)


def yolo_demo():
    model_path = "/root/yolo_onnx_models/yolo11m.engine"
    image_path = "/root/My_project/DemoTensorRT/images/bus.jpg"
    model = YOLO(model_path)
    results = model(image_path)
    print(results[0].boxes)


def main():
    convert_tensorRT_engine()
    # yolo_demo()


if __name__ == "__main__":
    main()
