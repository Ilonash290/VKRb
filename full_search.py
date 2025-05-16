import random
import math
import time as tm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import seaborn as sns

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = {}
    matrix = []
    reading_matrix = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("dist_matrix:"):
            reading_matrix = True
            continue
        if reading_matrix:
            matrix.append(list(map(float, line.split())))
        else:
            key, value = line.split(":")
            data[key.strip()] = float(value.strip())
    
    data['dist_matrix'] = np.array(matrix)
    return data

def greedy_route(courier_order_indices, warehouse_index, dist_matrix):
    if not courier_order_indices:
        return [], 0
    route = []
    current = warehouse_index
    total_distance = 0
    unvisited = list(courier_order_indices)
    while unvisited:
        next_order = min(unvisited, key=lambda idx: dist_matrix[current][idx])
        route.append(next_order)
        total_distance += dist_matrix[current][next_order]
        current = next_order
        unvisited.remove(next_order)
    total_distance += dist_matrix[current][warehouse_index]
    return route, total_distance

def calculate_time(courier_order_indices, warehouse_index, dist_matrix, speed):
    route, total_distance = greedy_route(courier_order_indices, warehouse_index, dist_matrix)
    delivery_time = total_distance / speed
    return delivery_time, route

def can_assign_to_k(k, orders_indices, warehouse_index, dist_matrix, speed, work_time, max_orders_per_courier):
    assignment = [[] for _ in range(k)]
    def backtrack(remaining_orders):
        if not remaining_orders:
            return True
        order = remaining_orders[0]
        for courier in range(k):
            if len(assignment[courier]) < max_orders_per_courier:
                temp_assignment = assignment[courier] + [order]
                delivery_time, _ = calculate_time(temp_assignment, warehouse_index, dist_matrix, speed)
                if delivery_time <= work_time:
                    assignment[courier].append(order)
                    if backtrack(remaining_orders[1:]):
                        return True
                    assignment[courier].pop()
        return False
    if backtrack(orders_indices):
        return True, assignment
    else:
        return False, None

if __name__ == "__main__":
    file_path = "input_data.txt"
    data = read_input_file(file_path)
    speed = data['speed']
    work_time = data['work_time']
    dist_matrix = data['dist_matrix']
    m = dist_matrix.shape[0] - 1
    
    warehouse_index = 0
    orders_indices = list(range(1, m + 1))
    max_orders_per_courier = m
    
    start_time = tm.time()
    
    low = 1
    high = m
    min_k = None
    best_assignment = None
    
    while low <= high:
        mid = (low + high) // 2
        print(f"\nПроверка с {mid} коммивояжерами")
        success, assignment = can_assign_to_k(mid, orders_indices, warehouse_index, dist_matrix, speed, work_time, max_orders_per_courier)
        if success:
            min_k = mid
            best_assignment = assignment
            high = mid - 1
            end_time = tm.time()
            execution_time = end_time - start_time
            print(f"Время выполнения программы({mid}): {execution_time:.2f} секунд")
        else:
            low = mid + 1
            end_time = tm.time()
            execution_time = end_time - start_time
            print(f"Время выполнения программы({mid}): {execution_time:.2f} секунд")
    
    if min_k is not None:
        actual_used_couriers = sum(1 for ca in best_assignment if ca)
        print(f"\nМинимальное количество коммивояжеров: {actual_used_couriers}")
        best_routes = []
        for ca in best_assignment:
            if ca:
                _, route = calculate_time(ca, warehouse_index, dist_matrix, speed)
                best_routes.append(route)
            else:
                best_routes.append([])
        courier_number = 1
        for ca, route in zip(best_assignment, best_routes):
            if ca:
                delivery_time, _ = calculate_time(ca, warehouse_index, dist_matrix, speed)
                route_str = ' -> '.join([f'Город {idx}' for idx in route])
                print(f"Коммивояжер {courier_number}: склад -> {route_str} -> склад, время: {delivery_time:.2f} ч")
                courier_number += 1
        end_time = tm.time()
        execution_time = end_time - start_time
        print(f"Время выполнения программы: {execution_time:.2f} секунд")
        
    else:
        print("Невозможно посетить все города с заданными параметрами.")