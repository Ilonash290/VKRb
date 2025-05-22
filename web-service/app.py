from flask import Flask, request, jsonify, send_file, render_template
from salesman_optimization import SalesmanOptimization
import os
import numpy as np
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return jsonify({"error": "Файл слишком большой. Максимальный размер файла — 200 МБ."}), 413

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        pop_size = request.form.get('pop_size')
        generations = request.form.get('generations')
        mutation_rate = request.form.get('mutation_rate')
        kwargs = {}
        if pop_size is not None:
            kwargs['pop_size'] = int(pop_size)
        if generations is not None:
            kwargs['generations'] = int(generations)
        if mutation_rate is not None:
            kwargs['mutation_rate'] = float(mutation_rate)

        optimization = SalesmanOptimization.from_file(file_path, **kwargs)
        used_salesmans, best_routes = optimization.find_min_salesmans()
        
        if used_salesmans is None:
            return jsonify({"error": "Не удалось найти решение с заданными параметрами. Попробуйте изменить значения параметров или ограничений"}), 500
        
        best_routes = [route for route in best_routes if route]

        plot_path = os.path.join(UPLOAD_FOLDER, 'routes.png')
        optimization.plot_routes(used_salesmans, best_routes, output_file=plot_path)
        
        result = {
            "used_salesmans": used_salesmans,
            "routes": best_routes
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot', methods=['GET'])
def get_plot():
    plot_path = os.path.join(UPLOAD_FOLDER, 'routes.png')
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    return jsonify({"error": "Plot not found"}), 404

@app.route('/manual_optimize', methods=['POST'])
def manual_optimize():
    try:
        speed = float(request.form['speed'])
        work_time = float(request.form['work_time'])
        dist_matrix_str = request.form['dist_matrix']
        dist_matrix = []
        for line in dist_matrix_str.split('\n'):
            line = line.strip()
            if line:
                row = list(map(float, line.split()))
                dist_matrix.append(row)
        dist_matrix = np.array(dist_matrix)

        pop_size = request.form.get('pop_size')
        generations = request.form.get('generations')
        mutation_rate = request.form.get('mutation_rate')
        kwargs = {}
        if pop_size is not None:
            kwargs['pop_size'] = int(pop_size)
        if generations is not None:
            kwargs['generations'] = int(generations)
        if mutation_rate is not None:
            kwargs['mutation_rate'] = float(mutation_rate)

        optimization = SalesmanOptimization(
            dist_matrix=dist_matrix,
            speed=speed,
            work_time=work_time,
            **kwargs
        )
        used_salesmans, best_routes = optimization.find_min_salesmans()

        if used_salesmans is None:
            return jsonify({"error": "Не удалось найти решение с заданными параметрами. Попробуйте изменить значения параметров или ограничений."}), 500

        best_routes = [route for route in best_routes if route]

        plot_path = os.path.join(UPLOAD_FOLDER, 'routes.png')
        optimization.plot_routes(used_salesmans, best_routes, output_file=plot_path)

        result = {
            "used_salesmans": used_salesmans,
            "routes": best_routes
        }
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": "Неверный формат ввода: " + str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
