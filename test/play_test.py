import cv2
import time
import os
import re
import numpy as np
import subprocess

# ------------------------------
# 配置参数
# ------------------------------
CONFIG = {
    "physical_size_cm": 15,
    "screen_size_inch": 14,
    "total_play_ms": 200,
    "final_pause_ms": 20,
    "aruco_dir": "./aruco_markers",
    "seq_order": [1, 2, 3, 4],
    "total_loop_count": 50,
    "loop_interval_sec": 1,
    "sync_signal_file": "./.aruco_test_start_signal",
    "main_screen_output": "XWAYLAND0",
    "portable_screen_output": "XWAYLAND4"
}

# ------------------------------
# 工具函数
# ------------------------------
def get_screen_info(output_name: str) -> dict:
    """获取屏幕的分辨率、位置（x/y偏移）"""
    try:
        xrandr_output = subprocess.check_output(["xrandr"], text=True)
        lines = xrandr_output.split("\n")
        for line in lines:
            if output_name in line and "connected" in line:
                res_match = re.search(r"(\d+)x(\d+)", line)
                pos_match = re.search(r"\+(\d+)\+(\d+)", line)
                if res_match and pos_match:
                    return {
                        "width": int(res_match.group(1)),
                        "height": int(res_match.group(2)),
                        "x": int(pos_match.group(1)),
                        "y": int(pos_match.group(2))
                    }
    except:
        pass
    # 默认值
    if output_name == CONFIG["main_screen_output"]:
        return {"width":2560, "height":1600, "x":0, "y":0}
    else:
        return {"width":2160, "height":1440, "x":2560, "y":0}

def pixel_per_cm(screen_w: int, screen_h: int) -> float:
    return np.sqrt(screen_w**2 + screen_h**2) / (CONFIG["screen_size_inch"] * 2.54)

# ------------------------------
# 主播放逻辑
# ------------------------------
def main():
    # 1. 获取双屏信息
    main_screen = get_screen_info(CONFIG["main_screen_output"])
    portable_screen = get_screen_info(CONFIG["portable_screen_output"])
    
    print("="*60)
    print(f"📽️ Aruco 50次循环播放器（双屏精准定位版）")
    print(f"🖥️  主屏幕：{main_screen['width']}×{main_screen['height']}（位置：{main_screen['x']},{main_screen['y']}）")
    print(f"🖥️  便携屏：{portable_screen['width']}×{portable_screen['height']}（位置：{portable_screen['x']},{portable_screen['y']}）")
    print("="*60)
    print("提示：按 'q' 可提前退出")

    # 2. 环境变量
    os.environ["OPENCV_UI_MODE"] = "GTK"
    os.environ["GDK_BACKEND"] = "x11"
    os.environ.pop("QT_QPA_PLATFORM", None)

    # 3. 清理历史信号
    if os.path.exists(CONFIG["sync_signal_file"]):
        os.remove(CONFIG["sync_signal_file"])
        print(f"🗑️  清理历史同步信号")

    # 4. 加载Aruco码
    aruco_paths = []
    for seq in CONFIG["seq_order"]:
        found = False
        for filename in os.listdir(CONFIG["aruco_dir"]):
            if f"seq{seq}.png" in filename:
                aruco_paths.append(os.path.join(CONFIG["aruco_dir"], filename))
                found = True
                break
        if not found:
            print(f"❌ 未找到seq{seq}的Aruco码，请检查目录！")
            return

    if len(aruco_paths) != 4:
        print(f"❌ 只找到{len(aruco_paths)}/4个Aruco码，播放失败！")
        return
    print(f"✅ 成功加载4个Aruco码：{[os.path.basename(p) for p in aruco_paths]}")

    # 5. 尺寸计算
    target_size = int(CONFIG["physical_size_cm"] * pixel_per_cm(portable_screen["width"], portable_screen["height"]))
    target_size = min(target_size, portable_screen["width"], portable_screen["height"])
    x_center = max(0, (portable_screen["width"] - target_size) // 2)
    y_center = max(0, (portable_screen["height"] - target_size) // 2)
    print(f"📏 便携屏播放尺寸：{target_size}px×{target_size}px（15cm×15cm）")

    # 6. 创建窗口到便携屏
    cv2.namedWindow("Aruco Player", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.moveWindow("Aruco Player", portable_screen["x"], portable_screen["y"])
    cv2.resizeWindow("Aruco Player", portable_screen["width"], portable_screen["height"])
    cv2.setWindowProperty("Aruco Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 7. 初始空白帧
    blank_bg = np.ones((portable_screen["height"], portable_screen["width"], 3), dtype=np.uint8) * 255
    cv2.imshow("Aruco Player", blank_bg)
    cv2.waitKey(500)

    # 8. 3秒缓冲+发送信号
    print("\n⏳ 准备就绪！3秒后开始播放...")
    for i in range(3, 0, -1):
        print(f"🔔 {i}秒后启动...")
        time.sleep(1)
    
    with open(CONFIG["sync_signal_file"], "w") as f:
        f.write("start")
    print(f"📶 已发送开始信号")

    # 9. 循环播放
    for loop_idx in range(1, CONFIG["total_loop_count"] + 1):
        print(f"\n🔄 第{loop_idx:02d}/{CONFIG['total_loop_count']}次循环")
        
        single_sec = CONFIG["total_play_ms"] / 1000 / len(aruco_paths)
        start_single = time.time()
        
        for i, path in enumerate(aruco_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"⚠️  无法加载{os.path.basename(path)}")
                continue
            
            # 缩放并居中显示
            img_resized = cv2.resize(img, (target_size, target_size), cv2.INTER_LINEAR)
            frame = blank_bg.copy()
            frame[y_center:y_center+target_size, x_center:x_center+target_size] = img_resized
            
            # 显示
            cv2.imshow("Aruco Player", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 手动退出")
                cv2.destroyAllWindows()
                return
            
            # 打印信息+控制时间
            bin_str = os.path.basename(path).split("_")[1]
            print(f"▶️  {bin_str}（{i+1}/{len(aruco_paths)}）")
            elapsed = time.time() - start_single - i*single_sec
            time.sleep(max(0, single_sec - elapsed))
        
        # 停留+间隔
        if CONFIG["final_pause_ms"] > 0:
            print(f"⏸️  停留{CONFIG['final_pause_ms']}ms...")
            time.sleep(CONFIG["final_pause_ms"] / 1000)
        
        if loop_idx < CONFIG["total_loop_count"]:
            print(f"⏳ 间隔{CONFIG['loop_interval_sec']}秒...")
            time.sleep(CONFIG["loop_interval_sec"])

    # 清理+退出
    if os.path.exists(CONFIG["sync_signal_file"]):
        os.remove(CONFIG["sync_signal_file"])
    
    print("\n" + "="*60)
    print(f"🎉 完成{CONFIG['total_loop_count']}次循环播放！")
    print("="*60)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
