from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


# 定义预测API接口
@app.route('/')
def predict():
    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug=True)




