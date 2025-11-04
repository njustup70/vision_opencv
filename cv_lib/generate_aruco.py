import cv2
import os
from PIL import Image  # 用于设置图像DPI信息

def generate_aruco_by_binary(
    binary_str,  # 10位二进制字符串
    dict_type="DICT_7X7_1000",
    physical_size_cm=15,  # 固定为15cm×15cm
    dpi=300,  # 明确定义DPI参数（默认300）
    save_dir="./new_aruco_markers"
):
    """生成固定尺寸的Aruco码，确保DPI参数正确定义"""
    # 校验二进制格式
    if len(binary_str) != 10 or not all(c in "01" for c in binary_str):
        print(f"错误：二进制字符串必须是10位01组合，输入为{binary_str}")
        return
    
    # 转换二进制为ID
    marker_id = int(binary_str, 2)
    if marker_id > 999:
        print(f"错误：ID={marker_id}超过DICT_7X7_1000的最大支持值（999）")
        return
    
    # 解析码序号
    first_8bit = binary_str[:8]
    last_2bit = binary_str[8:]
    code_seq = None
    if first_8bit[:2] == "11":
        code_seq = 1
    elif first_8bit[:2] == "00":
        code_seq = 2
    elif first_8bit[:2] == "01":
        code_seq = 3
    elif first_8bit[:2] == "10":
        code_seq = 4
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算像素尺寸
    marker_size = int(physical_size_cm * dpi / 2.54)
    
    # 生成Aruco码
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_type))
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, borderBits=1)
    
    # 保存图像并嵌入DPI信息
    save_path = os.path.join(save_dir, f"aruco_bin_{binary_str}_id_{marker_id}_seq_{code_seq}_15cm.png")
    try:
        img = Image.fromarray(marker_img)
        img.save(save_path, dpi=(dpi, dpi))
        print(f"已生成：{save_path}（包含DPI={dpi}信息）")
    except ImportError:
        cv2.imwrite(save_path, marker_img)
        print(f"已生成：{save_path}（未安装PIL，需手动按{dpi}DPI显示/打印）")
    
    print(f"  物理尺寸：15cm×15cm，像素尺寸：{marker_size}×{marker_size}，ID：{marker_id}\n")

if __name__ == "__main__":
    # 示例生成
    generate_aruco_by_binary("1100100100")
    generate_aruco_by_binary("0001001000")
    generate_aruco_by_binary("0110001100")
    generate_aruco_by_binary("1001100000")
