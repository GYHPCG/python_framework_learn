from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)


# 定义预测API接口
@app.route('/')
def predict():
    return "hello flask"

if __name__ == '__main__':
    app.run(debug=True)




