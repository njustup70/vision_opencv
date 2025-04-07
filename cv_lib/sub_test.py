# sub.py
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")  # 连接发布者
socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息

while True:
    message = socket.recv_string()
    print("收到:", message)
