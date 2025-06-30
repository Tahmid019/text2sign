from flask import Flask, request, jsonify

from gestDetect import sl2t

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    value = data.get('value')
    
    if value is None:
        return jsonify({'error': 'No value provided'}), 400

    sl2t()
    
    result = f'Completed. Closing...'

    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")