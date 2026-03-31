import rclpy
import struct
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String, Float32
import time
import sys
import numpy as np
from rclpy.guard_condition import GuardCondition
from queue import Queue
from get_dispose_serial.myserial import AsyncSerial_t 
class SerialDisposeNode(Node):
    def __init__(self):
        super().__init__('serial_dispose_node')
        self.serial = AsyncSerial_t(
            port="/dev/tnt1",
            baudrate=115200,
        )
        self.serial.startListening(lambda data: self.serial_callback(data))
        self.publisher_ = self.create_publisher(Vector3Stamped, 'lidar_position', 10)
        self.publisher_odom = self.create_publisher(Vector3Stamped, 'odom_data', 10)
        self.subscription_cmd_vel = self.create_subscription(Twist,'cmd_vel',self.cmd_vel_callback,10)
        self.subscription_mao = self.create_subscription(Float32,'/small_board_pose/confidence',self.cmd_maotou_callback,10)
        self.rx_queue = Queue()
        self.guard = self.create_guard_condition(self.process_rx_queue)
        self.out_first_frame = b'\xFA'
        self.out_frame_list = {
            "joy_cmd": {
            "id": b'\xA0',
            "fmt": "<fff"
        },
            "maotou":{
            "id": b'\xB1',
            "fmt": "<f"
        },
}
        self.in_first_frame = b'\xFF'
        self.in_frame_list = {
            b'\xAA': {
                "name": "odom_frame",
                "dispose": self.dispose_odom,
                "send": self.publisher_odom,
                "fmt": "<fff"
            },
            
        }   
    def cmd_vel_callback(self, msg):
        linear_x = msg.linear.x
        linear_y = msg.linear.y
        angular_z = msg.angular.z
        data = struct.pack(self.out_frame_list["joy_cmd"]["fmt"], linear_x,linear_y, angular_z)
        send_joy = self.out_first_frame + self.out_frame_list["joy_cmd"]["id"] + data
        self.serial.write(send_joy)
    def cmd_maotou_callback(self, msg):
        confidence = float(msg.data)
        print("CALLBACK ENTERED")
        data = struct.pack(self.out_frame_list["maotou"]["fmt"], confidence)
        send_maotou = self.out_first_frame + self.out_frame_list["maotou"]["id"] + data
        self.serial.write(send_maotou)
        print(send_maotou)
    def serial_callback(self, data): 
        print("RX:", data.hex())
        self.rx_queue.put(data)
        self.guard.trigger()
    def process_rx_queue(self):
        while not self.rx_queue.empty():
            data = self.rx_queue.get()
            if len(data) < 2:
                continue
            if data[0:1] != self.in_first_frame:
                continue
            frame_id = data[1:2]
            frame_data = self.in_frame_list.get(frame_id)
            if frame_data is None:
                continue
            else :
                if len(data) < 2 + struct.calcsize(frame_data["fmt"]):
                    continue
                get_data = frame_data["dispose"](data[2:2 + struct.calcsize(frame_data["fmt"])],frame_data["fmt"])
                frame_data["send"].publish(get_data)

    def dispose_odom(self, data, fmt):
        x, y, z = struct.unpack(fmt, data)
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = x
        msg.vector.y = y
        msg.vector.z = z
        return msg
        
def main(args=None):
    rclpy.init(args=args)
    serial_dispose_node = SerialDisposeNode()
    rclpy.spin(serial_dispose_node)
    serial_dispose_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()