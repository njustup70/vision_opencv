import cv2
import time
import os
import re
import numpy as np
import multiprocessing
from threading import Thread, Lock
from queue import Queue
from aruco_lib import Aruco

# ------------------------------
# 1. 配置参数
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
    "screen_size_inch": 16
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
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        # 启动后台线程
        self.thread = Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()
        print(f"📂 异步保存线程启动，保存目录：{save_dir}")

    def _worker(self):
        """后台工作线程：持续处理保存任务"""
        while self.is_running:
            try:
                frame, marker_id = self.queue.get(timeout=1)
                timestamp = time.strftime("%Y%m%d_%H%M%S_%f", time.localtime())[:-3]
                save_path = os.path.join(
                    self.save_dir,
                    f"detected_{timestamp}_ID{marker_id}.png"
                )
                cv2.imwrite(save_path, frame)
                print(f"💾 保存识别结果：{os.path.basename(save_path)}")
                self.queue.task_done()
            except:
                continue

    def add_save_task(self, frame: np.ndarray, marker_ids: list):
        """添加保存任务到队列"""
        if not marker_ids:
            return
        
        with self.lock:
            for mid in marker_ids:
                if mid not in self.saved_ids:
                    frame_copy = frame.copy()
                    if not self.queue.full():
                        self.queue.put((frame_copy, mid))
                        self.saved_ids.add(mid)

    def stop(self):
        """停止线程"""
        self.is_running = True
        self.queue.join()
        print(f"📥 异步保存线程停止，共保存 {len(self.saved_ids)} 个结果")

# ------------------------------
# 3. 核心业务逻辑
# ------------------------------
class KFSArucoService:
    def __init__(self):
        self.aruco_detector = Aruco(aruco_type=CONFIG["aruco_type"], if_draw=True)
        self.marker_binaries = {1: None, 2: None, 3: None, 4: None}
        self.pos_states = {i: "未知" for i in range(1, 13)}
        self.unrecognized_counters = {i: 0 for i in range(1, 13)}
        self.async_saver = AsyncSaveThread(CONFIG["detected_save_dir"])

    def encode_states(self, input_states: list) -> list:
        """12个状态 → 4个10位二进制串"""
        if len(input_states) != 12:
            raise ValueError("必须输入12个状态")
        
        valid_states = CONFIG["reverse_status_map"].keys()
        for s in input_states:
            if s not in valid_states:
                raise ValueError(f"无效状态：{s}（有效：{list(valid_states)}）")
        
        groups = [input_states[i*3:(i+1)*3] for i in range(4)]
        prefixes = ["11", "00", "01", "10"]
        return [prefix + "".join(CONFIG["reverse_status_map"][s] for s in g) + "00" 
                for prefix, g in zip(prefixes, groups)]

    def decode_markers(self, marker_ids: list) -> dict:
        """Marker ID列表 → 12个位置状态"""
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
                bit_str = bin_data[i*2:(i+1)*2] if len(bin_data)>=i*2+2 else ""
                if len(bit_str) == 2:
                    self.pos_states[pos] = CONFIG["status_map"][bit_str]
                    self.unrecognized_counters[pos] = 0
        
        for pos in range(1,13):
            self.unrecognized_counters[pos] += 1
            if self.unrecognized_counters[pos] >= CONFIG["stable_threshold"]:
                self.pos_states[pos] = "未知"
        
        return self.pos_states

    def save_detected_marker(self, frame: np.ndarray, marker_ids: list):
        """调用异步保存"""
        self.async_saver.add_save_task(frame, marker_ids)

# ------------------------------
# 4. 工具函数
# ------------------------------
def generate_aruco_by_lib(binary_str: str) -> str:
    """调用库生成Aruco码"""
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
    print(f"📁 生成Aruco：{os.path.basename(save_path)}")
    return save_path

def get_screen_res() -> tuple:
    """获取屏幕分辨率"""
    try:
        output = os.popen("xrandr").read()
        match = re.search(r"current (\d+) x (\d+)", output)
        return (int(match.group(1)), int(match.group(2))) if match else (1920, 1080)
    except:
        return (1920, 1080)

def pixel_per_cm(screen_w: int, screen_h: int) -> float:
    """像素密度（像素/厘米）"""
    diagonal_px = np.sqrt(screen_w**2 + screen_h**2)
    diagonal_cm = CONFIG["screen_size_inch"] * 2.54
    return diagonal_px / diagonal_cm

def play_aruco(aruco_paths: list):
    """独立进程播放Aruco"""
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    screen_w, screen_h = get_screen_res()
    target_size = int(CONFIG["physical_size_cm"] * pixel_per_cm(screen_w, screen_h))
    imgs = []
    for path in aruco_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 无法加载：{path}")
            return
        imgs.append(cv2.resize(img, (target_size, target_size)))
    
    window_name = "Aruco Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, 0, 0)
    blank_bg = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255
    x, y = (screen_w - target_size)//2, (screen_h - target_size)//2
    cv2.imshow(window_name, blank_bg)
    cv2.waitKey(5)
    
    total_sec = CONFIG["total_play_ms"] / 1000
    single_sec = total_sec / len(imgs)
    print(f"🎬 播放：{len(imgs)}个码，总有效时长{total_sec*1000:.0f}ms（每个{single_sec*1000:.0f}ms）")
    
    start_total = time.time()
    for i, img in enumerate(imgs):
        frame = blank_bg.copy()
        frame[y:y+target_size, x:x+target_size] = img
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
        
        bin_str = os.path.basename(aruco_paths[i]).split("_")[1]
        print(f"▶️  {bin_str}（{i+1}/{len(imgs)}）")
        
        start_single = time.time()
        while time.time() - start_single < single_sec:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
    
    print(f"⏱️  实际有效时长：{(time.time()-start_total)*1000:.0f}ms")
    if CONFIG["final_pause_ms"] > 0:
        print(f"⏸️  停留{CONFIG['final_pause_ms']}ms...")
        start_pause = time.time()
        while time.time() - start_pause < CONFIG["final_pause_ms"]/1000:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("🗑️  播放结束")

# ------------------------------
# 5. 主流程
# ------------------------------
def main():
    print("="*60)
    print(f"有效状态：['空', 'R1', 'R2', '假']")
    print("输入：12个状态空格分隔 | 退出：按'q'")
    print(f"识别结果保存目录：{CONFIG['detected_save_dir']}")
    print("="*60)
 
    service = KFSArucoService()
    
    # 1. 输入状态并编码
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
    
    # 2. 生成Aruco
    print("\n🔧 生成Aruco码...")
    try:
        aruco_paths = [generate_aruco_by_lib(bin_str) for bin_str in binary_strs]
    except Exception as e:
        print(f"❌ 生成失败：{e}")
        return
    
    # 3. 启动摄像头
    print("\n📹 启动摄像头...")
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        print(f"❌ 摄像头启动失败！请检查 camera_index={CONFIG['camera_index']}")
        return
    
    # 配置摄像头参数
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["cam_w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["cam_h"])
    cap.set(cv2.CAP_PROP_FPS, CONFIG["cam_fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 验证参数是否生效
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"⚠️  摄像头实际参数：{actual_w:.0f}×{actual_h:.0f} @ {actual_fps:.0f}FPS")
    if actual_fps < CONFIG["cam_fps"] * 0.8:
        print(f"⚠️  警告：摄像头不支持{CONFIG['cam_fps']}FPS，实际仅{actual_fps:.0f}FPS")
        CONFIG["cam_fps"] = int(actual_fps)
    
    # 摄像头预热
    for _ in range(10):
        cap.read()
    print(f"✅ 摄像头就绪：{CONFIG['cam_w']}×{CONFIG['cam_h']} @ {CONFIG['cam_fps']}FPS")
    
    # 4. 启动播放进程
    print("\n📽️  启动Aruco播放...")
    play_process = multiprocessing.Process(target=play_aruco, args=(aruco_paths,))
    play_process.start()
    
    # 5. 识别主循环
    print("\n🔍 开始识别（按'q'退出）")
    last_result = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 摄像头读取失败")
            time.sleep(0.1)
            continue
        
        # 检测并绘制
        detected_frame = service.aruco_detector.detect_image(
            input_data=frame,
            aruco_type=CONFIG["aruco_type"],
            if_draw=True
        )
        # 提取Marker ID
        marker_results = service.aruco_detector.update(frame)
        marker_ids = [res["id"] for res in marker_results]
        
        # 解码并打印结果
        pos_states = service.decode_markers(marker_ids)
        current_result = str([(pos, pos_states[pos]) for pos in range(1,13)])
        if current_result != last_result:
            last_result = current_result
            print("\n🔍 解码结果：")
            for pos in range(1,13):
                print(f"  位置{pos}：{pos_states[pos]}")
        
        # 异步保存
        service.save_detected_marker(detected_frame, marker_ids)
        
        # 显示画面
        cv2.imshow("Detection", detected_frame)
        
        # 退出逻辑
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 退出中... 等待保存任务完成...")
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
    print("✅ 程序完全退出")

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    multiprocessing.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 程序中断")
    except Exception as e:
        print(f"\n❌ 异常退出：{e}")