import cv2
import time
import os
import re
import numpy as np
import multiprocessing
from threading import Thread, Lock
from queue import Queue
import queue
import sys

cv_lib_dir = "/home/fishros/commun_ws/src/vision_opencv"
sys.path.append(os.path.abspath(cv_lib_dir))
from cv_lib.aruco_lib import Aruco

# ------------------------------
# 1. 简化配置参数
# ------------------------------
CONFIG = {
    "aruco_type": "DICT_7X7_1000",
    "physical_size_cm": 15,
    "dpi": 300,
    "save_dir": "./aruco_markers",
    "detected_save_dir": "./detected_aruco",
    "status_map": {"00": "空", "01": "R1KFS", "10": "R2KFS", "11": "假KFS"},
    "reverse_status_map": {"空": "00", "R1": "01", "R2": "10", "假": "11"},
    "camera_index": 10,
    "cam_w": 320, "cam_h": 240, "cam_fps": 120,
    "stable_threshold": 3,
    "total_play_ms": 200,
    "final_pause_ms": 200,
    "screen_size_inch": 16,
    "exposure_level": 30
}

# ------------------------------
# 2. 异步保存线程
# ------------------------------
class AsyncSaveThread:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.queue = Queue(maxsize=10)
        self.saved_ids = set()
        self.lock = Lock()
        self.is_running = True
        
        os.makedirs(save_dir, exist_ok=True)
        self.thread = Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()
        print(f"📂 识别结果保存目录：{save_dir}")

    def _worker(self):
        while self.is_running:
            try:
                frame, marker_id = self.queue.get(timeout=1)
                timestamp = time.strftime("%Y%m%d_%H%M%S_%f", time.localtime())[:-3]
                save_path = os.path.join(
                    self.save_dir,
                    f"detected_{timestamp}_ID{marker_id}.png"
                )
                cv2.imwrite(save_path, frame)
                print(f"💾 保存：{os.path.basename(save_path)}")
                self.queue.task_done()
            except queue.Empty:
                continue

    def add_save_task(self, frame: np.ndarray, marker_ids: list):
        if not marker_ids:
            return
        
        with self.lock:
            for mid in marker_ids:
                if mid not in self.saved_ids and not self.queue.full():
                    self.queue.put((frame.copy(), mid))
                    self.saved_ids.add(mid)

    def stop(self):
        self.is_running = False
        self.queue.join()
        print(f"📥 共保存 {len(self.saved_ids)} 个识别结果")

# ------------------------------
# 3. 核心业务逻辑
# ------------------------------
class KFSArucoService:
    def __init__(self):
        self.aruco_detector = Aruco(aruco_type=CONFIG["aruco_type"], if_draw=True)
        self.marker_binaries = {1: None, 2: None, 3: None, 4: None}
        self.pos_states = ["未知"] * 13
        self.unrecognized_counters = [0] * 13
        self.async_saver = AsyncSaveThread(CONFIG["detected_save_dir"])

    def encode_states(self, input_states: list) -> list:
        if len(input_states) != 12:
            raise ValueError("必须输入12个状态")
        
        valid_states = CONFIG["reverse_status_map"].keys()
        for s in input_states:
            if s not in valid_states:
                raise ValueError(f"无效状态：{s}（有效：{list(valid_states)}）")
        
        groups = [input_states[i*3:(i+1)*3] for i in range(4)]
        prefixes = ["11", "00", "01", "10"]
        return [f"{prefix}{''.join(CONFIG['reverse_status_map'][s] for s in g)}00" 
                for prefix, g in zip(prefixes, groups)]

    def decode_markers(self, marker_ids: list) -> list:
        for mid in marker_ids:
            try:
                bin8 = bin(mid)[2:].zfill(10)[:8]
                seq = {"11":1, "00":2, "01":3, "10":4}.get(bin8[:2])
                if seq:
                    self.marker_binaries[seq] = bin8
            except Exception as e:
                print(f"⚠️ 解析ID={mid}失败：{e}")
        
        for seq in range(1,5):
            if not self.marker_binaries[seq]:
                continue
            bin_data = self.marker_binaries[seq][2:]
            for i in range(3):
                pos = (seq-1)*3 + 1 + i
                if len(bin_data) >= i*2+2:
                    self.pos_states[pos] = CONFIG["status_map"][bin_data[i*2:(i+1)*2]]
                    self.unrecognized_counters[pos] = 0
        
        for pos in range(1,13):
            self.unrecognized_counters[pos] += 1
            if self.unrecognized_counters[pos] >= CONFIG["stable_threshold"]:
                self.pos_states[pos] = "未知"
        
        return self.pos_states

# ------------------------------
# 4. 工具函数
# ------------------------------
def generate_aruco(binary_str: str) -> str:
    """简化函数名，删除冗余注释"""
    marker_id = int(binary_str, 2)
    if marker_id > 999:
        raise ValueError(f"ID={marker_id}超过{CONFIG['aruco_type']}上限（999）")
    
    marker_size = int(CONFIG["physical_size_cm"] * CONFIG["dpi"] / 2.54)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    seq = {"11":1, "00":2, "01":3, "10":4}[binary_str[:2]]
    save_path = os.path.join(CONFIG["save_dir"], f"aruco_{binary_str}_id{marker_id}_seq{seq}.png")
    
    aruco = Aruco()
    aruco.aruco_maker(
        aruco_type=Aruco.ARUCO_DICT[CONFIG["aruco_type"]],
        ids=marker_id,
        pix=marker_size,
        path=save_path
    )
    print(f"📁 生成：{os.path.basename(save_path)}")
    return save_path

def get_screen_res() -> tuple:
    try:
        if match := re.search(r"current (\d+) x (\d+)", os.popen("xrandr").read()):
            return (int(match.group(1)), int(match.group(2)))
    except:
        pass
    return (1920, 1080)

def pixel_per_cm(screen_w: int, screen_h: int) -> float:
    return np.sqrt(screen_w**2 + screen_h**2) / (CONFIG["screen_size_inch"] * 2.54)

def play_aruco(aruco_paths: list):
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    screen_w, screen_h = get_screen_res()
    target_size = int(CONFIG["physical_size_cm"] * pixel_per_cm(screen_w, screen_h))
    
    # 图像加载逻辑
    imgs = [cv2.resize(cv2.imread(path), (target_size, target_size)) 
            for path in aruco_paths if cv2.imread(path) is not None]
    if len(imgs) != len(aruco_paths):
        print("❌ 部分Aruco图像加载失败")
        return
    
    # 窗口初始化
    cv2.namedWindow("Aruco Player", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Aruco Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    blank_bg = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255
    x, y = (screen_w - target_size)//2, (screen_h - target_size)//2
    cv2.imshow("Aruco Player", blank_bg)
    cv2.waitKey(5)
    
    # 播放计时
    single_sec = CONFIG["total_play_ms"] / 1000 / len(imgs)
    print(f"🎬 播放：{len(imgs)}个码（每个{single_sec*1000:.0f}ms）")
    
    start_total = time.time()
    for i, img in enumerate(imgs):
        frame = blank_bg.copy()
        frame[y:y+target_size, x:x+target_size] = img
        cv2.imshow("Aruco Player", frame)
        cv2.waitKey(1)
        
        bin_str = os.path.basename(aruco_paths[i]).split("_")[1]
        print(f"▶️  {bin_str}（{i+1}/{len(imgs)}）")
        
        time.sleep(max(0, single_sec - 0.001))
    
    print(f"⏱️  实际时长：{(time.time()-start_total)*1000:.0f}ms")
    if CONFIG["final_pause_ms"] > 0:
        print(f"⏸️  停留{CONFIG['final_pause_ms']}ms...")
        time.sleep(CONFIG["final_pause_ms"]/1000)
    
    cv2.destroyAllWindows()
    print("🗑️  播放结束")

# ------------------------------
# 5. 主流程
# ------------------------------
def main():
    print("="*60)
    print("📋 KFS-Aruco 系统")
    print(f"有效状态：['空', 'R1', 'R2', '假'] | 输入：12个状态空格分隔 | 退出：q")
    print("="*60)
 
    service = KFSArucoService()
    
    while True:
        input_states = input("请输入12个位置状态：").strip().split()
        if len(input_states) == 12:
            try:
                binary_strs = service.encode_states(input_states)
                break
            except ValueError as e:
                print(f"❌ {e}")
        else:
            print(f"❌ 需12个状态（当前{len(input_states)}个）")
    
    # 生成Aruco
    print("\n🔧 生成Aruco码...")
    try:
        aruco_paths = [generate_aruco(bin_str) for bin_str in binary_strs]
    except Exception as e:
        print(f"❌ 生成失败：{e}")
        return
    
    # 启动摄像头
    print("\n📹 启动摄像头...")
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        print(f"❌ 摄像头启动失败！检查 camera_index={CONFIG['camera_index']}")
        return
    
    # 简化摄像头参数配置
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, CONFIG["exposure_level"])
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["cam_w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["cam_h"])
    cap.set(cv2.CAP_PROP_FPS, CONFIG["cam_fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 简化参数验证
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps < CONFIG["cam_fps"] * 0.8:
        print(f"⚠️  摄像头不支持{CONFIG['cam_fps']}FPS，实际{actual_fps:.0f}FPS")
        CONFIG["cam_fps"] = int(actual_fps)
    
    # 摄像头预热
    for _ in range(10):
        cap.read()
    print(f"✅ 摄像头就绪：{CONFIG['cam_w']}×{CONFIG['cam_h']} @ {CONFIG['cam_fps']}FPS")
    
    # 启动播放进程
    print("\n📽️  启动Aruco播放...")
    play_process = multiprocessing.Process(target=play_aruco, args=(aruco_paths,))
    play_process.start()
    
    # 识别主循环
    print("\n🔍 开始识别（按'q'退出）")
    last_result = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 检测并绘制
        detected_frame = service.aruco_detector.detect_image(frame, CONFIG["aruco_type"], if_draw=True)
        marker_results = service.aruco_detector.update(detected_frame)
        marker_ids = [res["id"] for res in marker_results]
        
        # 解码并简化打印
        current_pos = service.decode_markers(marker_ids)
        current_result = str(current_pos[1:13])
        if current_result != last_result:
            last_result = current_result
            print("\n🔍 解码结果：")
            print(f"  位置1-12：{current_pos[1]}/{current_pos[2]}/{current_pos[3]}/{current_pos[4]}/{current_pos[5]}/{current_pos[6]}/{current_pos[7]}/{current_pos[8]}/{current_pos[9]}/{current_pos[10]}/{current_pos[11]}/{current_pos[12]}")
        
        # 异步保存
        service.async_saver.add_save_task(detected_frame, marker_ids)
        
        # 显示画面
        cv2.imshow("Detection", cv2.resize(detected_frame, (480, 360)))
        
        # 退出逻辑
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 退出中...")
            service.async_saver.stop()
            if play_process.is_alive():
                play_process.terminate()
            play_process.join()
            break
        
        # 播放结束提示
        if not play_process.is_alive() and not hasattr(main, "play_ended"):
            main.play_ended = True
            print("\n📢 播放完成！可继续识别已生成的Aruco图像")
    
    # 资源清理
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 程序退出")

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    multiprocessing.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 程序中断")
    except Exception as e:
        print(f"\n❌ 异常退出：{e}")