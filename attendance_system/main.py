import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from datetime import datetime

from typer.colors import BLACK

from face_align import align_face_similarity
from face_recognition import FaceRecognition
from database import AttendanceDB

from PIL import Image, ImageDraw, ImageFont

def cv2_put_text_chinese(img, text, position, font_path, font_size=32, color=(0,255,0)):
    """
    在 OpenCV 图像上绘制中文
    :param img: OpenCV 图像 (BGR)
    :param text: 要绘制的文本
    :param position: (x, y) 文本左下角坐标
    :param font_path: 中文字体文件路径
    :param font_size: 字体大小
    :param color: 颜色 (B, G, R)
    :return: 绘制后的图像（直接修改原图）
    """
    # 将 OpenCV 的 BGR 转换为 PIL 的 RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    # 绘制文本
    draw.text(position, text, font=font, fill=color[::-1])  # color 转为 RGB
    # 转换回 OpenCV BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 初始化 MediaPipe FaceLandmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=3,#最多同时检查3张人脸
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.FaceLandmarker.create_from_options(options)

# 初始化人脸识别和数据库
recognizer = FaceRecognition(model_path='models/mobilefacenet.tflite')
db = AttendanceDB()

# 加载已有特征库
database = db.get_all_users()
print(f"已加载 {len(database)} 个用户")

# 控制变量
last_attendance = {}          # 用于时间间隔打卡（若不需要可忽略）
ATTENDANCE_INTERVAL = 5       # 5秒内不可重复打卡

def check_and_record(name):
    """检查是否允许打卡（基于时间间隔）"""
    now = time.time()
    last = last_attendance.get(name, 0)
    if now - last < ATTENDANCE_INTERVAL:
        print(f"{name} 打卡间隔过短，跳过")
        return False
    if db.add_attendance(name):   # 需要此方法在 database.py 中实现
        last_attendance[name] = now
        print(f"{name} 打卡成功，时间 {datetime.now()}")
        return True
    else:
        print(f"打卡失败：{name} 不存在？")
        return False

# ---------- 性能优化参数 ----------
DETECT_EVERY_N_FRAMES = 3   # 每3帧检测一次（可根据实际调整）
frame_counter = 0

# 用于存储上一次检测结果的变量
last_result = {
    'has_face': False,
    'faces': []
}

# 记录上一次识别到的姓名（用于打卡防重复）
last_detected_name = None

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 'q' 退出")
print("按 'r' 注册新用户（多角度采集）")
print("按 't' 查看今日打卡记录")
print("按 'd' 删除已注册用户")
print("按 's' 保存当前对齐图像（调试用）")
print("按 'u' 查看已注册用户列表")

def print_help():
    print("\n========== 可用操作 ==========")
    print("按 'q' 退出")
    print("按 'r' 注册新用户（多角度采集）")
    print("按 't' 查看今日打卡记录")
    print("按 'u' 查看已注册用户列表")
    print("按 'd' 删除已注册用户")
    print("按 's' 保存当前对齐图像（调试用）")
    print("================================\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_counter += 1
    should_detect = (frame_counter % DETECT_EVERY_N_FRAMES == 0)

    # ---------- 检测帧：执行完整流程 ----------
    if should_detect:
        # 转换颜色空间
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            current_faces = []
            for face_landmarks in detection_result.face_landmarks:
                # 对齐
                aligned = align_face_similarity(frame, face_landmarks,
                                                output_size=(112, 112),
                                                eye_distance_ratio=0.35,
                                                eye_y_ratio=0.38,
                                                nose_y_ratio=0.68)
                # 提取特征
                feature = recognizer.extract_feature(aligned)
                # 识别身份
                name, sim = recognizer.identify(feature, database, threshold=0.6)
                current_faces.append({
                    'landmarks': face_landmarks,
                    'aligned': aligned,
                    'name': name,
                    'similarity': sim
                })
            last_result['faces'] = current_faces
            last_result['has_face'] = True

            # 打卡处理
            for face in current_faces:
                name = face['name']
                if name:
                    if db.add_attendance_if_not_today(name):
                        print(f"{name} 打卡成功")
        else:
            last_result['has_face'] = False
            last_result['faces'] = []

    # ---------- 非检测帧：直接使用上次结果 ----------
    # 显示识别信息
    # 在主循环的显示部分
    if last_result['has_face']:
        for face in last_result['faces']:
            name = face['name']
            sim = face['similarity']
            display_name = name if name else "Unknown"
            display_sim = f"{sim:.2f}" if sim is not None else ""

            landmarks = face['landmarks']
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x_min = int(min(xs) * frame.shape[1])
            x_max = int(max(xs) * frame.shape[1])
            y_min = int(min(ys) * frame.shape[0])
            y_max = int(max(ys) * frame.shape[0])

            # 绘制矩形框
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # 绘制中文姓名
            text = f"{display_name} {display_sim}"
            frame = cv2_put_text_chinese(frame, text, (x_min, y_min - 10),
                                         font_path='fonts/simhei.ttf',  # 请确保路径正确
                                         font_size=20,
                                         color=(0, 255, 0) if name else (0, 0, 255))
        # 显示第一个人的对齐图像（可选）
        if last_result['faces']:
            cv2.imshow('Aligned Face', last_result['faces'][0]['aligned'])
    else:
        if cv2.getWindowProperty('Aligned Face', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Aligned Face')
    # 显示原始画面
    cv2.imshow('Attendance System', frame)

    # ---------- 按键处理 ----------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('r'):
        # ========== 多角度注册流程（修正版） ==========
        print("\n========== 注册新用户 ==========")
        print("请面对摄像头，缓慢转动头部（上下左右），保持人脸在画面中央")
        print("按 'c' 开始采集（3秒后自动开始）...")
        # 等待用户按键开始
        start_wait = False
        while True:
            ret, frame_temp = cap.read()
            if not ret:
                break
            cv2.imshow('Attendance System', frame_temp)
            key_wait = cv2.waitKey(10) & 0xFF
            if key_wait == ord('c'):
                start_wait = True
                break
            elif key_wait == ord('q'):
                break
        if not start_wait:
            continue
        # 倒计时
        for i in range(3, 0, -1):
            print(f"{i}...")
            ret, frame_temp = cap.read()
            if ret:
                cv2.putText(frame_temp, str(i), (frame_temp.shape[1] // 2, frame_temp.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
                cv2.imshow('Attendance System', frame_temp)
                cv2.waitKey(1000)
            else:
                time.sleep(1)
        collected_features = []
        num_samples = 8  # 采集张数
        sample_interval = 2  # 采集间隔（秒）
        window_name = 'Aligning Registration'  # 统一窗口名
        print(f"开始采集 {num_samples} 张人脸，请缓慢改变角度...")
        for i in range(num_samples):
            ret, frame_sample = cap.read()
            if not ret:
                print("摄像头读取失败")

                break
            # 显示实时画面
            cv2.imshow('Attendance System', frame_sample)
            # 人脸检测与对齐
            rgb_sample = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2RGB)
            mp_image_sample = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_sample)
            result_sample = detector.detect(mp_image_sample)
            if result_sample.face_landmarks:
                landmarks = result_sample.face_landmarks[0]
                aligned_sample = align_face_similarity(

                    frame_sample, landmarks,

                    output_size=(112, 112),

                    eye_distance_ratio=0.35,

                    eye_y_ratio=0.38,

                    nose_y_ratio=0.68

                )
                feat = recognizer.extract_feature(aligned_sample)
                collected_features.append(feat)
                # 显示对齐图像
                cv2.imshow(window_name, aligned_sample)
                print(f"已采集 {len(collected_features)}/{num_samples} 张")
            else:
                print("未检测到人脸，跳过此帧")
            # 等待间隔，同时刷新窗口
            start_wait = time.time()
            while time.time() - start_wait < sample_interval:
                ret, frame_wait = cap.read()
                if ret:
                    cv2.imshow('Attendance System', frame_wait)
                cv2.waitKey(10)
        # 安全关闭对齐窗口
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(window_name)
        if len(collected_features) < 3:
            print("有效人脸不足3张，注册失败")
        # 注册成功后的部分
        if len(collected_features) >= 3:
            avg_feature = np.mean(collected_features, axis=0)
            name_input = input("请输入姓名：").strip()
            if name_input:
                db.register_user(name_input, avg_feature)
                database = db.get_all_users()
                last_detected_name = None
                print(f"用户 {name_input} 注册成功，共采集 {len(collected_features)} 张")

                # ---------- 恢复焦点 ----------
                cv2.setWindowProperty('Attendance System', cv2.WND_PROP_TOPMOST, 1)  # 窗口置顶
                print("请点击窗口并按空格键继续...")
                while True:
                    key_wait = cv2.waitKey(0) & 0xFF
                    if key_wait == ord(' '):
                        break
                cv2.setWindowProperty('Attendance System', cv2.WND_PROP_TOPMOST, 0)  # 取消置顶
                print_help()
                # --------------------------
            else:
                    print("姓名不能为空，注册取消")
        else:
                print("有效人脸不足3张，注册失败")

    elif key == ord('t'):
        records = db.get_today_attendance()
        if records:
            print("今日打卡记录：")
            for name, ts in records:
                print(f"{name}: {ts}")
        else:
            print("今日暂无打卡记录")
        print_help()

    elif key == ord('u'):
        users = db.get_all_user_names()
        if users:
            print("\n========== 已注册用户 ==========")
            for idx, name in enumerate(users, 1):
                print(f"{idx}. {name}")
            print("================================")
        else:
            print("暂无已注册用户")
        print_help()


    elif key == ord('d'):
        name_input = input("请输入要删除的用户姓名：").strip()
        if name_input:
            if db.delete_user(name_input):
                database = db.get_all_users()
                last_detected_name = None
                print(f"用户 {name_input} 已删除")
            else:
                print(f"用户 {name_input} 不存在")
        else:
            print("姓名不能为空")
        # 恢复焦点
        cv2.setWindowProperty('Attendance System', cv2.WND_PROP_TOPMOST, 1)
        print("请点击窗口并按空格键继续...")
        while True:
            key_wait = cv2.waitKey(0) & 0xFF
            if key_wait == ord(' '):
                break
        cv2.setWindowProperty('Attendance System', cv2.WND_PROP_TOPMOST, 0)
        print_help()

    elif key == ord('s'):
     if last_result['has_face'] and last_result['faces']:
        filename = f"saved_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, last_result['faces'][0]['aligned'])
        print(f"已保存对齐图像: {filename}")
     else:
        print("未检测到人脸，无法保存")

print_help()
# 释放资源
cap.release()
detector.close()
cv2.destroyAllWindows()