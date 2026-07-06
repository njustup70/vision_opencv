"""异步串口读写库 (从 spear/myserial.py 迁入 spear_vision 包)"""

import asyncio
import threading

import serial


class AsyncSerial_t:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._callback = None
        self._wait_time = 0.01
        self._raw_data = b''
        self._loop: asyncio.events.AbstractEventLoop
        self._thread = None
        self.last_len = 0
        self.data_queue = asyncio.Queue()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        asyncio.run_coroutine_threadsafe(self._connect_serial(), self._loop)
        asyncio.run_coroutine_threadsafe(self.__read(), self._loop)
        asyncio.run_coroutine_threadsafe(self.datahandle(), self._loop)

    async def _connect_serial(self):
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
        self._wait_time = wait_time
        if callback:
            self._callback = callback

    async def __read(self):
        while True:
            await asyncio.sleep(self._wait_time)
            if not self._serial or not self._serial.is_open:
                print("\033[91m[WARNING] Serial disconnected, retrying...\033[0m")
                self._serial = None
                await asyncio.sleep(1)
                continue
            try:
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
        while True:
            frame = await self.data_queue.get()
            if self.data_queue.qsize() > 10:
                while self.data_queue.qsize() != 10:
                    self.data_queue.get_nowait()
            if self._callback:
                try:
                    self._callback(frame)
                except Exception as e:
                    print(f"\033[91m[WARNING] Callback error: {e}\033[0m")

    def write(self, input_data: bytes) -> None:
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

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
