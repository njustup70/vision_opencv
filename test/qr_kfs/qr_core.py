import qrcode
import cv2
import os

class QRCoder:
    """二维码生成与解码"""
    
    STATUS_MAP = {"空": "00", "R1": "01", "R2": "10", "假": "11"}
    REVERSE_MAP = {v: k for k, v in STATUS_MAP.items()}
    RESERVE_BITS = "00000000"
    
    @classmethod
    def encode(cls, states, size_cm, dpi, save_dir="./qr_codes"):
        """编码状态为二维码"""
        if len(states) != 12:
            raise ValueError("需要12个状态")
        
        for state in states:
            if state not in cls.STATUS_MAP:
                raise ValueError(f"无效状态: {state}")
        
        binary = ''.join(cls.STATUS_MAP[s] for s in states) + cls.RESERVE_BITS
        hex_str = hex(int(binary, 2))[2:].zfill(8)
        
        target_pixels = int(round(size_cm * dpi / 2.54))
        
        module_count = 25 + 2 * 4
        box_size = max(1, target_pixels // module_count)
        
        # 生成QR码
        qr = qrcode.QRCode(
            version=2, 
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=box_size,
            border=2
        )
        qr.add_data(hex_str)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        if img.size[0] != target_pixels:
            img = img.resize((target_pixels, target_pixels), resample=0)
        
        os.makedirs(save_dir, exist_ok=True)
        filename = f"qr_{hex_str}.png"
        path = os.path.join(save_dir, filename)
        
        img.save(path, dpi=(dpi, dpi))
        
        img_cv = cv2.imread(path)
        if img_cv is not None:
            h, w = img_cv.shape[:2]
            actual_size = w * 2.54 / dpi
            print(f"✅ 二维码尺寸: {w}x{h}像素 ({actual_size:.1f}cm / 目标{size_cm}cm @ {dpi}DPI)")
            
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite(path, binary_img)
        
        return path, hex_str
    
    # 解码函数
    @classmethod
    def decode(cls, hex_str):
        if not hex_str or len(hex_str) != 8:
            return None
        
        try:
            binary = bin(int(hex_str, 16))[2:].zfill(32)
            state_bits = binary[:24]
            
            states = []
            for i in range(0, 24, 2):
                bits = state_bits[i:i+2]
                states.append(cls.REVERSE_MAP.get(bits, "未知"))
            
            return states
        except:
            return None
        
# if __name__ == "__main__":
    # 测试编码
    # states = ["空", "R1", "R2", "假", "空", "空", "R1", "R2", "假", "空", "R1", "R2"]
    # path, hex_str = QRCoder.encode(states, size_cm=10, dpi=220)
    # print(f"生成文件: {path}")
    # print(f"十六进制: {hex_str}")
    
    # 测试解码
    # decoded = QRCoder.decode(hex_str)
    # print(f"解码结果: {decoded}")