import cv2
import numpy as np
import tensorflow as tf

class FaceRecognition:
    def __init__(self, model_path='models/mobilefacenet.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # 输入形状应为 (1,112,112,3)
        self.input_shape = self.input_details[0]['shape'][1:3]  # (112,112)
        self.feature_size = self.output_details[0]['shape'][1]  # 512

    def preprocess(self, image):
        """
        image: BGR numpy array (H,W,3) 应为对齐后的112x112图像
        返回 (1,112,112,3) 归一化到 [-1,1]
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        img = np.expand_dims(img, axis=0)
        return img

    def extract_feature(self, image):
        """
        输入对齐后的人脸图像 (112,112,3)，返回特征向量 (512,)
        """
        input_data = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        feature = self.interpreter.get_tensor(self.output_details[0]['index'])
        return feature[0]  # (512,)

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def identify(self, feature, database, threshold=0.6):
        """
        在数据库中查找匹配身份
        database: dict {name: feature_vector}
        返回 (name, similarity) 或 (None, None)
        """
        max_sim = -1
        identity = None
        for name, db_feature in database.items():
            sim = self.cosine_similarity(feature, db_feature)
            if sim > max_sim:
                max_sim = sim
                identity = name
        if max_sim >= threshold:
            return identity, max_sim
        else:
            return None, None