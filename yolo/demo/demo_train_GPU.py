import os
import sys
import torch
from comet_ml import Experiment  # 必须放在torch之前导入
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    BASE_DIR = Path("/home/Nagisa/yolo/YOLOv11")
    DATASETS_DIR = BASE_DIR / "datasets"
    COCO8_DIR = DATASETS_DIR / "coco8"
    WEIGHTS_DIR = BASE_DIR / "weights"
    
    for d in [DATASETS_DIR, COCO8_DIR, WEIGHTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    os.environ['ULTRALYTICS_DATASETS_DIR'] = str(DATASETS_DIR)
    os.environ['ULTRALYTICS_HOME'] = str(BASE_DIR)

    # ============ 2. 初始化Comet ============
    experiment = Experiment(
        api_key="8KC8bgeb0kIOi3A1dWmM8YDVj",
        project_name="yolotest",
        workspace="linfei-tian",
        auto_param_logging=False
    )

    # ============ 3. 设备检测 ============
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*40}\n使用设备: {'GPU' if device != 'cpu' else 'CPU'}\n{'='*40}")

    # ============ 4. 硬编码参数配置 ============
    hyper_params = {
        "model": "yolov8n.pt",  # 或您的自定义模型路径
        "data": str(COCO8_DIR / "coco8.yaml"),  # 保持原路径结构
        "epochs": 3,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "amp": True,
        "device": device,
        "project": str(BASE_DIR.name),
        "name": "production_run",
        "save_dir": str(BASE_DIR)
    }
    experiment.log_parameters(hyper_params)

    # ============ 5. 训练执行 ============
    try:
        # 初始化模型
        print(f"\n加载模型: {hyper_params['model']}")
        model = YOLO(hyper_params["model"])

        # 训练监控回调
        def log_metrics(trainer):
            experiment.log_metrics({
                "epoch": trainer.epoch,
                "train_loss": trainer.metrics.get("train/box_loss", 0),
                "val_map50": trainer.metrics.get("val/mAP_50", 0)
            }, step=trainer.epoch)
        model.add_callback("on_train_epoch_end", log_metrics)

        # 开始训练
        print(f"\n开始训练，使用数据集: {hyper_params['data']}")
        print(f"关键参数: epochs={hyper_params['epochs']}, batch={hyper_params['batch']}")
        results = model.train(**hyper_params)

        # 记录最终指标
        if hasattr(results, 'results_dict'):
            experiment.log_metrics({
                "final_map50": results.results_dict.get("metrics/mAP_50", 0),
                "final_map50-95": results.results_dict.get("metrics/mAP_50-95", 0)
            })

        # 上传最佳模型
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_model_path.exists():
            experiment.log_model("best_model", str(best_model_path))
        
        print("\n训练成功完成!")

    except Exception as e:
        experiment.log_other("error", str(e))
        print(f"\n{'!'*40}\n训练失败: {str(e)}\n{'!'*40}", file=sys.stderr)
        sys.exit(1)