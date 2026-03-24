import sqlite3

conn = sqlite3.connect(r'D:\PyCharm\PycharmProjects\PythonProject\attendance_system\data\attendance.db')
cursor = conn.cursor()

# 查看所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("数据库中的表：")
for table in tables:
    print(table[0])

# 如果你知道正确的表名，比如 'attendance'，则查询它
table_name = 'users'   # 换成你看到的实际表名
try:
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
except sqlite3.OperationalError as e:
    print(f"查询出错：{e}")

conn.close()