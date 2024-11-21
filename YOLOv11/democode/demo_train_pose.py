from comet_ml import Experiment
from ultralytics import YOLO

if __name__ == '__main__':
    # 初始化 Comet 实验
    experiment = Experiment(
        api_key="8xk8gip09dycRGJVl8tNbgozY",
        project_name="yolov11-pose-test",
        workspace="summerwen-lab"
    )

    # 记录超参数
    hyper_params = {
        "model": "YOLOv11/pretrained_models/yolo11n-pose.pt",
        "epochs": 200,
        "imgsz": 640,
        "batch": 24,
        "amp": True
    }
    experiment.log_parameters(hyper_params)

    # 加载模型
    model = YOLO("YOLOv11/pretrained_models/yolo11n-pose.pt")

    # 开始训练
    results = model.train(
        task="pose",
        data="YOLOv11/demo_train_pose.yaml",
        epochs=hyper_params["epochs"],
        imgsz=hyper_params["imgsz"],
        batch=hyper_params["batch"],
        workers=0,
        amp=hyper_params["amp"]
    )

    # 记录关键点检测的指标到 Comet
    experiment.log_metric("final_kpt_loss", results.metrics["kpt_loss"])
    experiment.log_metric("final_map50", results.metrics["mAP_50"])
    experiment.log_metric("final_map50-95", results.metrics["mAP_50-95"])

