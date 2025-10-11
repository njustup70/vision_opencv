from comet_ml import Experiment

from ultralytics import YOLO


if __name__ == '__main__':
    # 初始化 Comet 实验
    experiment = Experiment(
        api_key="8xk8gip09dycRGJVl8tNbgozY",
        project_name="yolov11-test",
        workspace="summerwen-lab"
    )

    # 记录超参数
    hyper_params = {
        "model": "yolo11n",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "amp": True
    }
    experiment.log_parameters(hyper_params)

    # 加载模型
    model = YOLO("yolo11n.pt")

    # 开始训练
    results = model.train(
        data="YOLOv11/demo_train.yaml",
        epochs=hyper_params["epochs"],
        imgsz=hyper_params["imgsz"],
        batch=hyper_params["batch"],
        workers=0,
        amp=hyper_params["amp"]
    )

    # 手动记录最终指标到 Comet
    experiment.log_metric("final_loss", results.metrics["box_loss"])
    experiment.log_metric("final_map50", results.metrics["mAP_50"])
