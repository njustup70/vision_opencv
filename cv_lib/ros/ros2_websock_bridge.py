'''
@brief: rosbridge简单例程,可以发送/chatter话题,接收/camera/color/image_raw话题
@doc :官方文档连接:https://roslibpy.readthedocs.io/en/latest/
@note : 需要安装roslibpy库,并且在ros2中安装rosbridge_suite
'''
import time
import cv2
import roslibpy
import base64
import numpy as np
def callback(message):
    # print(message)
    width = message['width']
    height = message['height']

    # ROS 通过 rosbridge 发送的 uint8[] 数据字段会变成 base64 编码的字符串
    data_str = message['data']  # 这是 str 类型
    data_bytes = base64.b64decode(data_str)  # 解码成 bytes

    # 转换为 numpy 数组
    img_array = np.frombuffer(data_bytes, dtype=np.uint8).reshape((height, width, 3))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    cv2.imshow("Image", img_bgr)
    cv2.waitKey(1)
client = roslibpy.Ros(host='localhost', port=9090)
client.run()

talker = roslibpy.Topic(client, '/chatter', 'std_msgs/String')
listener = roslibpy.Topic(client, '/camera/color/image_raw', 'sensor_msgs/Image')
listener.subscribe(callback)
while client.is_connected:
    talker.publish(roslibpy.Message({'data': 'Hello World!'}))
    print('Sending message...')
    time.sleep(1)

talker.unadvertise()

client.terminate()