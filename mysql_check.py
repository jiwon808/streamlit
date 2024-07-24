import mysql.connector

# MySQL 연결 설정
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='mysql'
)

# 커서 생성
cursor = conn.cursor()

# SQL 쿼리 실행
cursor.execute("SELECT * FROM user")

# 결과 가져오기
rows = cursor.fetchall()

# 결과 출력
for row in rows:
    print(row)

# 연결 종료
cursor.close()
conn.close()
