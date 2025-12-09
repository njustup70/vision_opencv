import cv2
import time
import os
import numpy as np
from threading import Thread, Lock
from queue import Queue
import queue
from aruco_lib import Aruco

# ------------------------------
# 配置参数
# ------------------------------
CONFIG = {
    "aruco_type": "DICT_7X7_1000",
    "detected_save_dir": "./detected_aruco",
    "status_map": {"00": "空", "01": "R1KFS", "10": "R2KFS", "11": "假KFS"},
    "reverse_status_map": {"空": "00", "R1": "01", "R2": "10", "假": "11"},
    "camera_index": 10,
    "cam_w": 640, "cam_h": 480, "cam_fps": 120,
    "stable_threshold": 3,
    "exposure_level": 30,
    "test_loop_count": 50,
    "player_single_loop_ms": 1200,
    "loop_timeout_ms": 1200,
    "sync_signal_file": "./.aruco_test_start_signal"
}

# ------------------------------
# 异步保存线程
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
# 核心识别+同步统计逻辑
# ------------------------------
class KFSArucoDetector:
    def __init__(self):
        self.aruco_detector = Aruco(aruco_type=CONFIG["aruco_type"], if_draw=True)
        self.marker_binaries = {1: None, 2: None, 3: None, 4: None}
        self.pos_states = ["未知"] * 13
        self.unrecognized_counters = [0] * 13
        self.async_saver = AsyncSaveThread(CONFIG["detected_save_dir"])
        
        # 统计变量
        self.success_count = 0
        self.current_loop = 0
        self.loop_recognized_seqs = set()
        self.loop_start_time = time.time()
        self.total_test_timeout_ms = CONFIG["test_loop_count"] * CONFIG["player_single_loop_ms"] + 10000
        self.is_test_started = False

    # 解码
    def decode_markers(self, marker_ids: list) -> list:
        for mid in marker_ids:
            try:
                bin8 = bin(mid)[2:].zfill(10)[:8]
                seq = {"11":1, "00":2, "01":3, "10":4}.get(bin8[:2])
                if seq:
                    self.marker_binaries[seq] = bin8
                    self.loop_recognized_seqs.add(seq)
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
    
    def check_start_signal(self):
        if not self.is_test_started and os.path.exists(CONFIG["sync_signal_file"]):
            self.is_test_started = True
            self.loop_start_time = time.time()  # 收到信号后，才开始计时
            print(f"\n📶 收到开始信号！开始统计50次循环...")
            return True
        return self.is_test_started

    # 精准判断循环完成
    def check_loop_complete(self):
        if not self.check_start_signal():
            return False
        current_time = time.time()
        loop_duration_ms = (current_time - self.loop_start_time) * 1000
        
        # 条件1：达到超时上限
        if loop_duration_ms >= CONFIG["loop_timeout_ms"]:
            self.switch_to_next_loop()
        
        # 条件2：循环数达到50次 或 总时长超上限（播放器已结束）
        total_duration_ms = (current_time - self.loop_start_time) * 1000
        if self.current_loop >= CONFIG["test_loop_count"] or total_duration_ms >= self.total_test_timeout_ms:
            self.print_summary()
            return True
        
        return False

    # 切换到下一个循环
    def switch_to_next_loop(self):
        self.current_loop += 1
        if len(self.loop_recognized_seqs) == 4:
            self.success_count += 1
            print(f"✅ 第{self.current_loop:02d}次循环：识别成功（4个码全识别）")
        else:
            print(f"❌ 第{self.current_loop:02d}次循环：识别失败（仅识别到{len(self.loop_recognized_seqs)}/4个码）")
        # 重置状态，准备下一个循环
        self.loop_recognized_seqs = set()
        self.loop_start_time = time.time()

    # 打印汇总报告
    def print_summary(self):
        print("\n" + "="*80)
        print(f"📊 50次通信成功率测试汇总（已同步播放器结束）")
        print(f"总循环次数：{min(self.current_loop, CONFIG['test_loop_count'])}次")
        print(f"成功次数：{self.success_count}次")
        print(f"失败次数：{min(self.current_loop, CONFIG['test_loop_count']) - self.success_count}次")
        if self.current_loop > 0:
            success_rate = self.success_count / min(self.current_loop, CONFIG['test_loop_count']) * 100
            print(f"通信成功率：{success_rate:.2f}%")
        else:
            print(f"通信成功率：0.00%")
        print("="*80)

    # 启动识别
    def start_detect(self):
        print("\n📹 启动摄像头...")
        cap = cv2.VideoCapture(CONFIG["camera_index"])
        if not cap.isOpened():
            print(f"❌ 摄像头启动失败！检查 camera_index={CONFIG['camera_index']}")
            return
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, CONFIG["exposure_level"])
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_GAIN, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["cam_w"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["cam_h"])
        cap.set(cv2.CAP_PROP_FPS, CONFIG["cam_fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps < CONFIG["cam_fps"] * 0.8:
            print(f"⚠️  摄像头不支持{CONFIG['cam_fps']}FPS，实际{actual_fps:.0f}FPS")
            CONFIG["cam_fps"] = int(actual_fps)
        
        for _ in range(10):
            cap.read()
        print(f"✅ 摄像头就绪：{CONFIG['cam_w']}×{CONFIG['cam_h']} @ {CONFIG['cam_fps']}FPS")
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Detection", cv2.WND_PROP_TOPMOST, 1)

        print(f"\n⌛ 等待播放器开始信号（信号文件：{CONFIG['sync_signal_file']}）...")
        print("提示：启动播放器后，会自动发送信号，无需手动操作")
        
        last_result = ""

        frame_count = 0  # 帧计数
        fps_start_time = time.time()  # 帧率统计起始时间
        fps_print_interval = 1.0  # 每1秒打印一次帧率

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= fps_print_interval:
                real_fps = frame_count / elapsed_time
                print(f"📊 实时帧率：{real_fps:.1f} fps (目标：{CONFIG['cam_fps']} fps)")
                frame_count = 0
                fps_start_time = time.time()
            
            detected_frame = self.aruco_detector.detect_image(frame, CONFIG["aruco_type"], if_draw=True)
            marker_results = self.aruco_detector.update(detected_frame)
            marker_ids = [res["id"] for res in marker_results]
            current_pos = self.decode_markers(marker_ids)
            
            current_result = str(current_pos[1:13])
            if current_result != last_result:
                last_result = current_result
                if self.is_test_started:
                    print(f"\n🔍 解码结果：")
                    print(f"  位置1-12：{current_pos[1]}/{current_pos[2]}/{current_pos[3]}/{current_pos[4]}/{current_pos[5]}/{current_pos[6]}/{current_pos[7]}/{current_pos[8]}/{current_pos[9]}/{current_pos[10]}/{current_pos[11]}/{current_pos[12]}")
                else:
                    print(f"\n🔍 等待信号中，当前识别结果：")
                    print(f"  位置1-12：{current_pos[1]}/{current_pos[2]}/{current_pos[3]}/{current_pos[4]}/{current_pos[5]}/{current_pos[6]}/{current_pos[7]}/{current_pos[8]}/{current_pos[9]}/{current_pos[10]}/{current_pos[11]}/{current_pos[12]}")

            self.async_saver.add_save_task(detected_frame, marker_ids)
            
            cv2.imshow("Detection", cv2.resize(detected_frame, (480, 360)))
            
            if self.check_loop_complete():
                self.async_saver.stop()
                cap.release()
                cv2.destroyAllWindows()
                return
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 手动退出识别...")
                self.print_summary()
                self.async_saver.stop()
                cap.release()
                cv2.destroyAllWindows()
                return

# ------------------------------
# 主函数
# ------------------------------
def main():
    print("="*60)
    print(f"🔍 Aruco持续识别")
    print(f"有效状态：['空', 'R1', 'R2', '假'] | 退出：按 'q'")
    print("="*60)
    detector = KFSArucoDetector()
    detector.start_detect()

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    main()
