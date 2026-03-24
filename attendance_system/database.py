import sqlite3
import numpy as np
import os
from datetime import datetime

class AttendanceDB:
    def __init__(self, db_path='data/attendance.db'):
        self.db_path = db_path
        # 确保 data 目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # 员工表
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                feature BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 打卡记录表
        c.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()

    def register_user(self, name, feature):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        feature_blob = feature.tobytes()
        local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            c.execute("INSERT INTO users (name, feature, created_at) VALUES (?, ?, ?)",
                      (name, feature_blob, local_time))
            conn.commit()
            user_id = c.lastrowid
            print(f"用户 {name} 注册成功，ID: {user_id}")
        except sqlite3.IntegrityError:
            print(f"用户 {name} 已存在")
        finally:
            conn.close()

    def get_all_users(self):
        """返回 {name: feature_vector} 字典"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT name, feature FROM users")
        rows = c.fetchall()
        conn.close()
        database = {}
        for name, blob in rows:
            # 将 BLOB 还原为 numpy 数组（假设特征维度为512）
            feature = np.frombuffer(blob, dtype=np.float32)
            database[name] = feature
        return database

    def add_attendance_if_not_today(self, name):
        """如果今日未打卡，则添加打卡记录（使用本地时间）"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE name=?", (name,))
        row = c.fetchone()
        if not row:
            conn.close()
            return False
        user_id = row[0]
        # 获取今天的日期（本地时间）
        today = datetime.now().strftime("%Y-%m-%d")
        c.execute("SELECT COUNT(*) FROM attendance WHERE user_id=? AND DATE(timestamp)=?", (user_id, today))
        count = c.fetchone()[0]
        if count == 0:
            local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO attendance (user_id, timestamp) VALUES (?, ?)", (user_id, local_time))
            conn.commit()
            conn.close()
            return True
        else:
            conn.close()
            return False

    def get_today_attendance(self):
        """获取今日打卡记录"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        c.execute('''
            SELECT u.name, a.timestamp 
            FROM attendance a 
            JOIN users u ON a.user_id = u.id 
            WHERE DATE(a.timestamp) = ?
        ''', (today,))
        rows = c.fetchall()
        conn.close()
        return rows

    def get_all_user_names(self):
        """返回所有注册用户的姓名列表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM users ORDER BY id")
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows]

    def delete_user(self, name):
        """
        删除指定用户及其所有打卡记录
        返回 True 表示删除成功，False 表示用户不存在或删除失败
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            # 先获取用户 ID
            c.execute("SELECT id FROM users WHERE name=?", (name,))
            row = c.fetchone()
            if not row:
                conn.close()
                return False
            user_id = row[0]
            # 删除该用户的打卡记录（可选，若需要保留历史可注释）
            c.execute("DELETE FROM attendance WHERE user_id=?", (user_id,))
            # 删除用户本身
            c.execute("DELETE FROM users WHERE id=?", (user_id,))
            conn.commit()
            print(f"用户 {name} 及其打卡记录已删除")
            return True
        except Exception as e:
            print(f"删除失败: {e}")
            return False
        finally:
            conn.close()