# 注意在使用205的那台工作站的时候 路径前面要去除"YOLOv11"

from comet_ml import Experiment
from ultralytics import YOLO

if __name__ == '__main__':
    experiment = Experiment(
        api_key="8xk8gip09dycRGJVl8tNbgozY",
        project_name="basketball-recognition",
        workspace="summerwen-lab"
    )

    hyper_params = {
        "model": "yolo11n",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "amp": True,
       # "patience": 20, # Early Stopping --> epoch-limit = 5
        "workers": 8
    }
    experiment.log_parameters(hyper_params)

    model = YOLO("yolo11n.pt")

    results = model.train(
        data="YOLOv11/democode/basketball_general.yaml",
        epochs=hyper_params["epochs"],
        imgsz=hyper_params["imgsz"],
        batch=hyper_params["batch"],
        workers=hyper_params["workers"],
        amp=hyper_params["amp"],
        patience=hyper_params["patience"]
    )

    experiment.log_metric("final_loss", results.metrics["box_loss"])
    experiment.log_metric("final_map50", results.metrics["mAP_50"])
