from light_preproc import preproc_hsv, clahe_on_v, get_red_blue_mask
import cv2
import numpy as np

def mask_3x3_ratio(mask_u8):
    # 统一成 0/1，避免 255 带来的除法麻烦
    m = (mask_u8 != 0).astype(np.uint8)

    H, W = m.shape
    # 只取能整除的区域（避免边界慢处理）
    H3 = (H // 3) * 3
    W3 = (W // 3) * 3
    m = m[:H3, :W3]

    # reshape 成 (3, H/3, 3, W/3)，然后对块内求和
    blocks = m.reshape(3, H3 // 3, 3, W3 // 3)
    counts = blocks.sum(axis=(1, 3))  # -> shape (3, 3)

    area = (H3 // 3) * (W3 // 3)
    ratios = counts / float(area)
    return ratios  # 3x3，每块 mask 占比

def check_red_blue(warped):
    '''
    检查拉正图像中的红蓝分布，返回3x3网格的结果
    :param warped: 拉正后的图像 :type:`np.ndarray`
    :return: 3x3网格结果，0=无，1=红，2=蓝 :type:`List[List[int]]`
    '''
    result = [[0 for i in range(3)] for j in range(3)]

    mask_red = get_red_blue_mask(warped, blue=False, red=True, kernel_size=(7,7))
    mask_blue = get_red_blue_mask(warped, blue=True, red=False, kernel_size=(7,7))
    result_red = mask_3x3_ratio(mask_red)
    result_blue = mask_3x3_ratio(mask_blue)

    thr = 0.4
    red_hit  = (result_red  >= thr)
    blue_hit = (result_blue >= thr)

    # 0/1/2 编码：红=1，蓝=2（红蓝都满足会变 3）
    result = red_hit.astype(np.uint8) + 2 * blue_hit.astype(np.uint8)
    return result.tolist()

def main():
    img_name = "restored_5.jpg"
    img_path = "restored_images/" + img_name

    img = cv2.imread(img_path)
    img_original = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv = preproc_hsv(hsv)
    hsv = clahe_on_v(hsv, clip=2.0, grid=(8,8))

    result = check_red_blue(hsv)

    # print("Red mask 3x3 ratios:")
    # print(result_red)
    # print("Blue mask 3x3 ratios:")
    # print(result_blue)
    print("Final result (0: none, 1: red, 2: blue):")
    print(np.array(result).reshape(3,3))
    cv2.imshow("original", img_original)
    # cv2.namedWindow("MaskB", cv2.WINDOW_NORMAL)
    # cv2.imshow("MaskB", mask_blue)
    # cv2.moveWindow("MaskB", 20, 600)
    # cv2.namedWindow("MaskR", cv2.WINDOW_NORMAL)
    # cv2.imshow("MaskR", mask_red)
    # cv2.moveWindow("MaskR", 600, 600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()