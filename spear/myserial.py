"""_summary_串口异步读写库
@Author: LiuXuanze(Elaina-rascal)
@Date: 2024-12-29
@Description 使用方法:
1. 接收: AsyncSerial_t("COM2", 115200) 创建一个串口对象, 然后调用 register_callback() 开始监听串口数据;
   串口数据到来时会调用 callback 函数, 如果不传入 callback, 则可自行从队列处理。
2. 发送: write 函数用于向串口写入数据(阻塞函数)。

例程:
serial = AsyncSerial_t("COM2", 115200)
serial.register_callback(lambda data: serial.write(data))
"""

import asyncio
import threading
import time

import serial


class AsyncSerial_t:
    def __init__(self, port, baudrate):
        """初始化异步串口。"""
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._callback = None
        self._wait_time = 0.01
        self._raw_data = b''
        self._connect_lock = asyncio.Lock()
        self._loop: asyncio.events.AbstractEventLoop
        self._thread = None
        self.last_len = 0
        self.data_queue = asyncio.Queue()

        # 异步定义
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # 在后台循环里跑串口连接管理和读循环
        asyncio.run_coroutine_threadsafe(self._connect_serial(), self._loop)
        asyncio.run_coroutine_threadsafe(self.__read(), self._loop)
        asyncio.run_coroutine_threadsafe(self.datahandle(), self._loop)

    async def _connect_serial(self):
        """尝试连接串口，如果失败则等待重试。"""
        while True:
            if self._serial is None:
                try:
                    self._serial = serial.Serial(self.port, self.baudrate, timeout=0)
                    print(f"\033[92m[INFO] Serial connected: {self.port}\033[0m")
                except serial.SerialException as e:
                    print(f"\033[91m[WARNING] Could not connect to serial port {self.port}: {e}\033[0m")
            await asyncio.sleep(1)

    def __del__(self):
        if self._serial and self._serial.is_open:
            self._serial.close()

    def register_callback(self, callback=None, wait_time=0.0001) -> None:
        """开始监听串口数据, 启动 read 协程。"""
        self._wait_time = wait_time
        if callback:
            self._callback = callback

    async def __read(self):
        """异步读取串口数据并调用回调。"""
        while True:
            await asyncio.sleep(self._wait_time)
            if not self._serial or not self._serial.is_open:
                print("\033[91m[WARNING] Serial disconnected, retrying...\033[0m")
                self._serial = None
                await asyncio.sleep(1)
                continue

            try:
                # 检查数据是否停止发送
                this_len = self._serial.in_waiting
                if self._serial.in_waiting > 0:
                    if this_len != self.last_len:
                        self.last_len = this_len
                    else:
                        self.last_len = 0
                        data = self._serial.read(self._serial.in_waiting)
                        self.data_queue.put_nowait(data)
                        continue
            except Exception:
                try:
                    if self._serial:
                        self._serial.close()
                except Exception:
                    pass
                self._serial = None
                await asyncio.sleep(1)

    async def datahandle(self):
        """处理数据队列中的数据。"""
        while True:
            frame = await self.data_queue.get()
            # 如果 data_queue 大于 10 就丢弃旧数据
            if self.data_queue.qsize() > 10:
                while self.data_queue.qsize() != 10:
                    self.data_queue.get_nowait()
            if self._callback:
                try:
                    self._callback(frame)
                except Exception as e:
                    print(f"\033[91m[WARNING] Callback error: {e}\033[0m")

    def getRawData(self) -> bytes:
        """获取串口接收的原始数据。"""
        return self._raw_data

    def write(self, input_data: bytes) -> None:
        """向串口写入数据(阻塞)，如果串口可用。"""
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(input_data)
            except Exception as e:
                print(f"\033[91m[WARNING] Serial error during write: {e}\033[0m")
                try:
                    if self._serial:
                        self._serial.close()
                except Exception:
                    pass
                self._serial = None
        else:
            print("\033[91m[WARNING] Cannot write, serial not connected.\033[0m")
            # time.sleep(1)

    def _run_loop(self):
        """后台线程中运行事件循环。"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def process_queue(self):
        """示例消费者：在主协程里调用。"""
        while True:
            frame = await self.data_queue.get()
            print(f"[PROCESS] Got frame: {frame}")


# 示例主函数
async def main_async() -> None:
    serial_ins = AsyncSerial_t('/dev/qinheng', 230400)
    serial_ins.register_callback(lambda data: print(f"Received: {data.decode()}"))
    while True:
        data = await asyncio.to_thread(input, 'Please input data: ')
        serial_ins.write(data.encode())
        await asyncio.sleep(0.05)


def main():
    serial_ins = AsyncSerial_t('/dev/ttyACM0', 115200)
    serial_ins.register_callback(lambda data: print(f"hex: {data.hex()}"))
    while True:
        serial_ins.write(b'Hello from AsyncSerial_t!\n')
        time.sleep(1)


if __name__ == '__main__':
    main()