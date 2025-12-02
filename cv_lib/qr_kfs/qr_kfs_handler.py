import cv2
import time
import os
import re
import numpy as np
import qrcode
import multiprocessing
from pyzbar.pyzbar import decode
from threading import Thread, Lock
from queue import Queue

# ------------------------------
# 1. 核心配置
# ------------------------------
CONFIG = {
    "STATUS_MAP": {"空": "00", "R1": "01", "R2": "10", "假": "11"},
    "REVERSE_STATUS_MAP": {"00": "空", "01": "R1", "10": "R2", "11": "假"},
    "RESERVE_BITS": "00000000",
    "PHYSICAL_SIZE_CM": 15,
    "DPI": 300,
    "SAVE_DIR": "./new_qr_kfs_markers",
    "DETECTED_SAVE_DIR": "./detected_qr_kfs",
    "CAMERA_INDEX": 10,
    "CAM_W": 640, "CAM_H": 480,
    "CAM_FPS": 60,
    "TOTAL_PLAY_MS": 200,
    "FINAL_PAUSE_MS": 200,
    "SCREEN_SIZE_INCH": 16,
    "STABLE_THRESHOLD": 1
}

# ------------------------------
# 2. 异步保存线程
# ------------------------------
class AsyncSaveThread:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.queue = Queue(maxsize=10)
        self.saved_data = set()
        self.lock = Lock()
        self.is_running = True
        
        os.makedirs(save_dir, exist_ok=True)
        self.thread = Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()
        print(f"📂 异步保存线程启动 → {save_dir}")

    def _worker(self):
        while self.is_running:
            try:
                frame, qr_data = self.queue.get(timeout=1)
                timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                save_path = os.path.join(
                    self.save_dir,
                    f"detected_{timestamp}_QR{qr_data[:8]}.png"
                )
                cv2.imwrite(save_path, frame)
                print(f"💾 保存识别结果 → {os.path.basename(save_path)}")
                self.queue.task_done()
            except:
                continue

    def add_save_task(self, frame: np.ndarray, qr_data: str):
        if not qr_data:
            return
        
        with self.lock:
            if qr_data not in self.saved_data:
                self.queue.put((frame.copy(), qr_data))
                self.saved_data.add(qr_data)

    def stop(self):
        self.is_running = False
        self.queue.join()
        print(f"📥 保存线程停止 → 共保存 {len(self.saved_data)} 个结果")

# ------------------------------
# 3. 核心业务逻辑
# ------------------------------
class KFSQRService:
    def __init__(self):
        self.qr_binaries = None
        self.pos_states = {i: "未知" for i in range(1, 13)}
        self.unrecognized_counters = {i: 0 for i in range(1, 13)}
        self.async_saver = AsyncSaveThread(CONFIG["DETECTED_SAVE_DIR"])
        self.has_detected = False
        self.last_print_time = 0
        self.play_ended = False

    # 生成QR码
    def generate_qr(self, input_states: list) -> str:
        if len(input_states) != 12:
            raise ValueError("必须输入12个状态")
        
        valid_states = CONFIG["STATUS_MAP"].keys()
        for s in input_states:
            if s not in valid_states:
                raise ValueError(f"无效状态：{s}（有效：{list(valid_states)}）")
        
        kfs_bin = "".join([CONFIG["STATUS_MAP"][s] for s in input_states])
        total_bin = kfs_bin + CONFIG["RESERVE_BITS"]
        total_hex = hex(int(total_bin, 2))[2:].zfill(8)
        
        marker_size = int(CONFIG["PHYSICAL_SIZE_CM"] * CONFIG["DPI"] / 2.54)
        qr = qrcode.QRCode(
            version=2,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=marker_size // 25,
            border=6
        )
        qr.add_data(total_hex)
        qr.make(fit=True)
        
        os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
        state_summary = "_".join([CONFIG["STATUS_MAP"][s] for s in input_states])[:10] + "..."
        save_path = os.path.join(
            CONFIG["SAVE_DIR"],
            f"qr_kfs_{state_summary}_reserve_{CONFIG['RESERVE_BITS']}_15cm_300dpi.png"
        )
        qr.make_image(fill_color="black", back_color="white").save(save_path, dpi=(CONFIG["DPI"], CONFIG["DPI"]))
        
        print(f"📁 生成QR码→ {os.path.basename(save_path)}")
        print(f"  编码信息：{kfs_bin} + {CONFIG['RESERVE_BITS']} → 十六进制：{total_hex}")
        return save_path

    # 解码
    def detect_and_decode(self, frame: np.ndarray) -> tuple:
        frame_copy = frame.copy()
        qr_data = None
        self.qr_binaries = None

        if self.play_ended:
            return frame_copy, qr_data

        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        qr_codes = decode(gray)
        if qr_codes:
            (x, y, w, h) = qr_codes[0].rect
            if w >= 30 and h >= 30:
                qr_data = qr_codes[0].data.decode("utf-8").strip()
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_copy, qr_data[:8], (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if qr_data and len(qr_data) == 8:
            try:
                total_bin = bin(int(qr_data, 16))[2:].zfill(32)
                self.qr_binaries = total_bin[:24]
                self.has_detected = True
            except Exception as e:
                print(f"⚠️ 解析QR数据失败：{e}")
                qr_data = None

        if self.qr_binaries:
            for i in range(12):
                pos = i + 1
                bit_str = self.qr_binaries[i*2:(i+1)*2] if len(self.qr_binaries)>=i*2+2 else ""
                if len(bit_str) == 2 and bit_str in CONFIG["REVERSE_STATUS_MAP"]:
                    self.pos_states[pos] = CONFIG["REVERSE_STATUS_MAP"][bit_str]
                    self.unrecognized_counters[pos] = 0
        else:
            for pos in range(1, 13):
                self.unrecognized_counters[pos] += 1
                if self.unrecognized_counters[pos] >= CONFIG["STABLE_THRESHOLD"]:
                    self.pos_states[pos] = "未知"

        if self.has_detected:
            current_time = time.time()
            if current_time - self.last_print_time > 0.5:
                self.last_print_time = current_time

        if qr_data:
            self.async_saver.add_save_task(frame_copy, qr_data)

        return frame_copy, qr_data

# ------------------------------
# 4. 工具函数
# ------------------------------
def get_screen_res() -> tuple:
    try:
        output = os.popen("xrandr").read()
        match = re.search(r"current (\d+) x (\d+)", output)
        return (int(match.group(1)), int(match.group(2))) if match else (1920, 1080)
    except:
        return (1920, 1080)

def pixel_per_cm(screen_w: int, screen_h: int) -> float:
    diagonal_px = np.sqrt(screen_w**2 + screen_h**2)
    diagonal_cm = CONFIG["SCREEN_SIZE_INCH"] * 2.54
    return diagonal_px / diagonal_cm

def play_qr(qr_paths: list):
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    screen_w, screen_h = get_screen_res()
    target_size = int(CONFIG["PHYSICAL_SIZE_CM"] * pixel_per_cm(screen_w, screen_h))
    
    imgs = []
    for path in qr_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 无法加载QR码：{path}")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imgs.append(cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST))
    
    window_name = "QR-KFS Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(window_name, 0, 0)
    
    blank_bg = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255
    x, y = (screen_w - target_size) // 2, (screen_h - target_size) // 2
    cv2.imshow(window_name, blank_bg)
    cv2.waitKey(5)
    
    total_sec = CONFIG["TOTAL_PLAY_MS"] / 1000
    single_sec = total_sec / len(imgs) if len(imgs) > 0 else 0
    print(f"🎬 播放配置：{len(imgs)}个QR码，总时长{total_sec*1000:.0f}ms（每个{single_sec*1000:.0f}ms）")
    
    start_total = time.time()
    for i, img in enumerate(imgs):
        frame = blank_bg.copy()
        frame[y:y+target_size, x:x+target_size] = img
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
        
        qr_name = os.path.basename(qr_paths[i]).split("_")[1]
        print(f"▶️  播放 {qr_name}（{i+1}/{len(imgs)}）")
        
        start_single = time.time()
        while time.time() - start_single < single_sec:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
    
    print(f"⏱️  实际播放时长：{(time.time()-start_total)*1000:.0f}ms")
    if CONFIG["FINAL_PAUSE_MS"] > 0:
        print(f"⏸️  停留{CONFIG['FINAL_PAUSE_MS']}ms...")
        start_pause = time.time()
        while time.time() - start_pause < CONFIG["FINAL_PAUSE_MS"]/1000:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("🗑️  播放结束")

# ------------------------------
# 5. 主流程
# ------------------------------
def main():
    print("="*60)
    print("📋 QR-KFS 一体化系统")
    print("="*60)
    print(f"支持状态：{list(CONFIG['STATUS_MAP'].keys())}")
    print("输入：12个状态空格分隔 | 退出：按'q'")
    print(f"生成目录：{CONFIG['SAVE_DIR']} | 识别保存目录：{CONFIG['DETECTED_SAVE_DIR']}")
    print("="*60)
    
    service = KFSQRService()
    
    while True:
        input_str = input("\n请输入12个位置状态：").strip()
        if input_str.lower() == 'q':
            return
        input_states = input_str.split()
        if len(input_states) == 12:
            try:
                qr_path = service.generate_qr(input_states)
                break
            except ValueError as e:
                print(f"❌ {e}")
        else:
            print(f"❌ 需输入12个状态（当前{len(input_states)}个）")
    
    print("\n📹 启动摄像头...")
    cap = cv2.VideoCapture(CONFIG["CAMERA_INDEX"], cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"❌ 摄像头启动失败 → 检查索引{CONFIG['CAMERA_INDEX']}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["CAM_W"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["CAM_H"])
    cap.set(cv2.CAP_PROP_FPS, CONFIG["CAM_FPS"])
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, 25)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    
    print("⏳ 摄像头预热中...")
    for _ in range(10):
        cap.read()
        time.sleep(0.02)
    print(f"✅ 摄像头就绪 → {CONFIG['CAM_W']}×{CONFIG['CAM_H']} @ {CONFIG['CAM_FPS']}FPS")
    
    print("\n📽️  启动QR码播放...")
    play_process = multiprocessing.Process(target=play_qr, args=([qr_path],))
    play_process.start()
    
    print("\n🔍 开始识别（按'q'退出）")
    last_result = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 摄像头读取失败")
            time.sleep(0.01)
            continue
        
        detected_frame, qr_data = service.detect_and_decode(frame)
        
        if service.has_detected:
            current_result = str([(pos, service.pos_states[pos]) for pos in range(1, 13)])
            if current_result != last_result:
                last_result = current_result
                print("\n🎉 成功识别！稳定解码结果：")
                for pos in range(1, 13):
                    print(f"  位置{pos}：{service.pos_states[pos]}")
        
        cv2.imshow("QR-KFS Detection", detected_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 退出中... 等待保存任务完成...")
            service.async_saver.stop()
            if play_process.is_alive():
                play_process.terminate()
            play_process.join()
            break
        
        if not play_process.is_alive() and not hasattr(main, "play_ended"):
            main.play_ended = True
            service.play_ended = True
            if not service.has_detected:
                print("\n📢 播放完成但未识别到！请调整摄像头角度/距离")
            else:
                print("\n📢 播放完成，已成功识别！")
    
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
        print(f"\n❌ 异常退出 → {e}")
        import traceback
        traceback.print_exc()
