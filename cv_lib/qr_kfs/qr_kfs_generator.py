import qrcode
import os

# 状态与二进制映射（与之前方案完全一致）
STATUS_TO_BIN = {
    "空": "00",
    "R1KFS": "01",
    "R2KFS": "10",
    "假KFS": "11"
}

def generate_qr_by_kfs_states(
    kfs_states,  # 列表：12个位置的KFS状态（"空"/"R1KFS"/"R2KFS"/"假KFS"）
    reserve_bits="00000000",  # 8位预留二进制（可自定义）
    physical_size_cm=15,  # 固定物理尺寸：15cm×15cm
    dpi=300,  # 固定DPI：300
    save_dir="./new_qr_kfs_markers"
):
    """
    按指定的KFS状态生成QR码
    """
    # --------------------------
    # 1. 严格参数校验
    # --------------------------
    # 校验KFS状态列表
    if len(kfs_states) != 12:
        print(f"错误：KFS状态必须是12个位置，输入为{len(kfs_states)}个")
        return
    for idx, state in enumerate(kfs_states, 1):
        if state not in STATUS_TO_BIN:
            print(f"错误：位置{idx}状态无效（{state}），仅支持：{list(STATUS_TO_BIN.keys())}")
            return
    # 校验预留位
    if len(reserve_bits) != 8 or not all(c in ["0", "1"] for c in reserve_bits):
        print(f"错误：预留位必须是8位二进制字符串，输入为{reserve_bits}")
        return

    # --------------------------
    # 2. 编码逻辑
    # --------------------------
    # 编码12个位置状态（24位二进制）
    kfs_bin = ""
    for state in kfs_states:
        kfs_bin += STATUS_TO_BIN[state]
    total_bin = kfs_bin + reserve_bits
    total_hex = hex(int(total_bin, 2))[2:].zfill(8)

    # --------------------------
    # 3. 计算像素尺寸（确保15cm物理尺寸）
    # --------------------------
    pixel_size = int(physical_size_cm * dpi / 2.54)  # cm转像素（1英寸=2.54cm）
    print(f"编码信息：")
    print(f"  - KFS状态：{kfs_states}")
    print(f"  - 24位KFS二进制：{kfs_bin}")
    print(f"  - 8位预留二进制：{reserve_bits}")
    print(f"  - 8位十六进制（QR存储）：{total_hex}")
    print(f"  - 物理尺寸：{physical_size_cm}cm×{physical_size_cm}cm，DPI：{dpi}，像素尺寸：{pixel_size}×{pixel_size}")

    # --------------------------
    # 4. 生成QR码（带容错，适配打印/屏幕）
    # --------------------------
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=pixel_size // 21,
        border=4,
    )
    qr.add_data(total_hex)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")

    # --------------------------
    # 5. 保存图像
    # --------------------------
    os.makedirs(save_dir, exist_ok=True)
    state_summary = "_".join([STATUS_TO_BIN[state] for state in kfs_states])[:10] + "..."
    save_path = os.path.join(
        save_dir,
        f"qr_kfs_{state_summary}_reserve_{reserve_bits}_15cm_300dpi.png"
    )
    try:
        qr_img.save(save_path, dpi=(dpi, dpi))
        print(f"✅ 已生成：{save_path}\n")
    except ImportError:
        qr_img.save(save_path)
        print(f"✅ 已生成：{save_path}（未安装PIL，打印时需手动选择300DPI）\n")

if __name__ == "__main__":
    # --------------------------
    # 直接在代码中指定生成的KFS状态（按需修改）
    # --------------------------
    # 位置1-12自定义状态，预留位默认00000000
    generate_qr_by_kfs_states(
        kfs_states=[
            "空", "R2KFS", "R1KFS",  # 位置1-3
            "R2KFS", "空", "R1KFS",  # 位置4-6
            "假KFS", "空", "R2KFS",  # 位置7-9
            "空", "R1KFS", "R2KFS"   # 位置10-12
        ]
    )
