import cv2
import numpy as np

def align_face_similarity(image, landmarks, output_size=(112, 112),
                          eye_distance_ratio=0.35,
                          eye_y_ratio=0.38,
                          nose_y_ratio=0.68):
    """
    使用相似变换对齐人脸（仅旋转、缩放、平移），避免仿射变换带来的剪切变形。
    基于左右眼中心点计算变换，可选使用鼻尖微调垂直平移。
    """
    # 关键点索引（MediaPipe FaceLandmarker 468点）
    left_eye_outer = landmarks[33]
    left_eye_inner = landmarks[133]
    right_eye_outer = landmarks[263]
    right_eye_inner = landmarks[362]
    nose_tip = landmarks[1]

    # 计算左右眼中心（内外眼角中点）
    left_eye_center = np.array([(left_eye_outer.x + left_eye_inner.x) / 2,
                                 (left_eye_outer.y + left_eye_inner.y) / 2])
    right_eye_center = np.array([(right_eye_outer.x + right_eye_inner.x) / 2,
                                  (right_eye_outer.y + right_eye_inner.y) / 2])
    nose = np.array([nose_tip.x, nose_tip.y])

    h, w = image.shape[:2]
    # 原始图像中的点（像素坐标）
    left_eye_pix = left_eye_center * np.array([w, h])
    right_eye_pix = right_eye_center * np.array([w, h])
    nose_pix = nose * np.array([w, h])

    # 1. 计算原始两眼中心、角度和距离
    eye_center_src = (left_eye_pix + right_eye_pix) / 2.0
    eye_angle_src = np.degrees(np.arctan2(right_eye_pix[1] - left_eye_pix[1],
                                          right_eye_pix[0] - left_eye_pix[0]))
    eye_dist_src = np.linalg.norm(right_eye_pix - left_eye_pix)

    # 2. 目标图像中的两眼中心、距离（目标距离由 output_size 和 eye_distance_ratio 决定）
    out_w, out_h = output_size
    eye_dist_dst = out_w * eye_distance_ratio
    eye_center_dst = np.array([out_w / 2.0, out_h * eye_y_ratio])

    # 3. 计算相似变换参数
    angle = eye_angle_src
    scale = eye_dist_dst / eye_dist_src

    # 构建旋转矩阵（绕原始图像两眼中心旋转）
    M_rot = cv2.getRotationMatrix2D(tuple(eye_center_src), angle, scale)

    # 4. 计算旋转缩放后两眼中心的位置
    eye_center_src_h = np.append(eye_center_src, [1])
    eye_center_rot = M_rot @ eye_center_src_h
    translation = eye_center_dst - eye_center_rot
    M_rot[0, 2] += translation[0]
    M_rot[1, 2] += translation[1]

    # 5. 可选：使用鼻尖微调垂直平移
    if nose_y_ratio is not None:
        nose_pix_h = np.append(nose_pix, [1])
        nose_rot = M_rot @ nose_pix_h
        nose_target_y = out_h * nose_y_ratio
        delta_y = nose_target_y - nose_rot[1]
        M_rot[1, 2] += delta_y

    # 6. 应用变换
    aligned = cv2.warpAffine(image, M_rot, output_size, flags=cv2.INTER_LINEAR)
    return aligned