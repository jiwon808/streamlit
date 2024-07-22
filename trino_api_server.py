from flask import Flask, request, jsonify
import trino

app = Flask(__name__)

# Trino 연결 설정
conn = trino.dbapi.connect(
    host='localhost',
    port=8080,
    user='root',
    catalog='mysql',
    schema='mysql'
)

@app.route('/execute_sql', methods=['POST'])
def execute_sql():
    query = request.json.get('query')
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    return jsonify(rows)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
