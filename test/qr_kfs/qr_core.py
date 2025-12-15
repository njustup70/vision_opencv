import qrcode
import cv2
import os

class QRCoder:
    """二维码生成与解码"""
    
    STATUS_MAP = {"空": "00", "R1": "01", "R2": "10", "假": "11"}
    REVERSE_MAP = {v: k for k, v in STATUS_MAP.items()}
    RESERVE_BITS = "00000000"
    
    @classmethod
    def encode(cls, states, size_cm=15, dpi=300, save_dir="./qr_codes"):
        """编码状态为二维码"""
        if len(states) != 12:
            raise ValueError("需要12个状态")
        
        # 验证状态有效性
        for state in states:
            if state not in cls.STATUS_MAP:
                raise ValueError(f"无效状态: {state}")
        
        # 生成二进制
        binary = ''.join(cls.STATUS_MAP[s] for s in states) + cls.RESERVE_BITS
        hex_str = hex(int(binary, 2))[2:].zfill(8)
        
        # 生成QR
        pixel_size = int(size_cm * dpi / 2.54)
        qr = qrcode.QRCode(
            version=2, 
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=pixel_size // 25,
            border=6
        )
        qr.add_data(hex_str)
        qr.make()
        
        # 保存
        os.makedirs(save_dir, exist_ok=True)
        filename = f"qr_{hex_str}.png"
        path = os.path.join(save_dir, filename)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(path, dpi=(dpi, dpi))
        
        # 二值化处理
        img_cv = cv2.imread(path)
        if img_cv is not None:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite(path, binary_img)
        
        return path, hex_str
    
    @classmethod
    def decode(cls, hex_str):
        """解码十六进制为状态"""
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