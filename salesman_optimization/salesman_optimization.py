import random
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
        if not line:  # Пропускаем пустые строки
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

class SalesmanOptimization:
    def __init__(self, dist_matrix, speed, work_time, pop_size=200, generations=300, mutation_rate=0.015, elite_size_ratio=0.1):
        """Инициализация оптимизатора посещения.
        
        Args:
            dist_matrix (np.ndarray): Матрица расстояний (m+1 x m+1), где 0 — склад.
            speed (float): Скорость коммивояжеров в км/ч.
            work_time (float): Время работы коммивояжеров в часах.
            pop_size (int): Размер популяции.
            generations (int): Количество поколений.
            mutation_rate (float): Вероятность мутации.
            elite_size_ratio (float): Доля элитных особей.
        """
        self.dist_matrix = np.array(dist_matrix)
        self.speed = speed
        self.work_time = work_time
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = int(pop_size * elite_size_ratio)
        self.m = self.dist_matrix.shape[0] - 1  # Количество городов
        self.warehouse_index = 0
        self.orders_indices = list(range(1, self.m + 1))
        self.population = None
        self.best_chrom = None
        self.best_score = None
        self.current_generation = 0

    @classmethod
    def from_file(cls, file_path):
        """Создание экземпляра класса из файла."""
        data = read_input_file(file_path)
        return cls(
            speed=data['speed'],
            work_time=data['work_time'],
            dist_matrix=data['dist_matrix']
        )

    def greedy_route(self, salesman_order_indices):
        """Жадный алгоритм построения маршрута для коммивояжера."""
        if not salesman_order_indices:
            return [], 0
        route = []
        current = self.warehouse_index
        total_distance = 0
        unvisited = list(salesman_order_indices)
        while unvisited:
            next_order = min(unvisited, key=lambda idx: self.dist_matrix[current][idx])
            route.append(next_order)
            total_distance += self.dist_matrix[current][next_order]
            current = next_order
            unvisited.remove(next_order)
        total_distance += self.dist_matrix[current][self.warehouse_index]
        return route, total_distance

    def calculate_time(self, salesman_order_indices):
        """Расчет времени посещения для коммивояжера."""
        route, total_distance = self.greedy_route(salesman_order_indices)
        delivery_time = total_distance / self.speed
        return delivery_time, route

    def fitness(self, chromosome, n):
        """Функция приспособленности."""
        salesman_assignments = [[] for _ in range(n)]
        for order_idx, salesman in enumerate(chromosome):
            salesman_assignments[salesman].append(self.orders_indices[order_idx])
        
        total_time_excess = 0
        used_salesman_count = sum(1 for ca in salesman_assignments if ca)
        total_time = 0
        
        for ca in salesman_assignments:
            if ca:
                delivery_time, _ = self.calculate_time(ca)
                total_time += delivery_time
                if delivery_time > self.work_time:
                    total_time_excess += (delivery_time - self.work_time)
        
        if total_time_excess > 0:
            return 10000 + 100 * total_time_excess + used_salesman_count
        return used_salesman_count + 0.1 * total_time

    def initialize_population(self, n):
        """Инициализация популяции."""
        self.population = [[random.randint(0, n-1) for _ in range(self.m)] for _ in range(self.pop_size)]

    def genetic_algorithm_step(self, n):
        """Один шаг генетического алгоритма."""
        if self.population is None:
            self.initialize_population(n)
        
        fitness_scores = [self.fitness(chrom, n) for chrom in self.population]
        self.best_score = min(fitness_scores)
        self.best_chrom = self.population[fitness_scores.index(self.best_score)]
        
        if self.best_score < 10000:
            return True
        
        sorted_population = sorted(zip(self.population, fitness_scores), key=lambda x: x[1])
        new_population = [x[0] for x in sorted_population[:self.elite_size]]
        
        while len(new_population) < self.pop_size:
            tournament = random.sample(list(zip(self.population, fitness_scores)), 3)
            winner = min(tournament, key=lambda x: x[1])[0]
            new_population.append(winner[:])
        
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size and random.random() < 0.8:
                if self.m > 2:
                    point1 = random.randint(1, self.m - 2)
                    point2 = random.randint(point1 + 1, self.m - 1)
                    parent1, parent2 = new_population[i], new_population[i + 1]
                    new_population[i] = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
                    new_population[i + 1] = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        for chrom in new_population:
            for j in range(len(chrom)):
                if random.random() < self.mutation_rate:
                    chrom[j] = random.randint(0, n-1)
        
        self.population = new_population
        self.current_generation += 1
        return False

    def find_min_salesmans(self):
        """Поиск минимального количества коммивояжеров."""
        low = 1
        high = self.m
        min_salesmans = None
        best_chrom = None
        
        while low <= high:
            mid = (low + high) // 2
            self.population = None
            self.current_generation = 0
            for _ in range(self.generations):
                success = self.genetic_algorithm_step(mid)
                if success:
                    min_salesmans = mid
                    best_chrom = self.best_chrom
                    high = mid - 1
                    break
            else:
                low = mid + 1
        
        if min_salesmans is not None:
            salesman_assignments = [[] for _ in range(min_salesmans)]
            for order_idx, salesman in enumerate(best_chrom):
                salesman_assignments[salesman].append(self.orders_indices[order_idx])
            
            actual_used_salesmans = sum(1 for ca in salesman_assignments if ca)
            
            best_routes = []
            for ca in salesman_assignments:
                if ca:
                    _, route = self.calculate_time(ca)
                    best_routes.append(route)
                else:
                    best_routes.append([])

            return actual_used_salesmans, best_routes
        else:
            return None, None

    def plot_routes(self, used_salesmans, best_routes, output_file='routes.png'):
        """Визуализация маршрутов коммивояжеров."""
        coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(self.dist_matrix)
        pos = {i: (coords[i, 0], coords[i, 1]) for i in range(self.m + 1)}
        
        G = nx.DiGraph()
        for i in range(self.m + 1):
            G.add_node(i)
            for j in range(self.m + 1):
                if i != j:
                    G.add_edge(i, j, weight=self.dist_matrix[i][j])
        
        plt.figure(figsize=(18, 15))
        colors = sns.color_palette("husl", used_salesmans)
        
        order_colors = {}
        for idx, route in enumerate(best_routes):
            if route:
                for order in route:
                    order_colors[order] = colors[idx]
        
        for idx, route in enumerate(best_routes):
            if route:
                full_route = [self.warehouse_index] + route + [self.warehouse_index]
                route_edges = [(full_route[j], full_route[j+1]) for j in range(len(full_route) - 1)]
                edge_colors = [colors[idx]] * len(route_edges)
                nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color=edge_colors,
                                       width=2, arrowstyle='-|>', arrowsize=30, label=f'Коммивояжер {idx+1}', arrows=True)
                route_edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in route_edges}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=route_edge_labels, font_color='black', font_size=16)
        
        nx.draw_networkx_nodes(G, pos, nodelist=[self.warehouse_index], node_color='black', node_size=400, node_shape='s')
        for order in self.orders_indices:
            color = order_colors.get(order, 'lightblue')
            nx.draw_networkx_nodes(G, pos, nodelist=[order], node_color=color, node_size=400)
        
        labels = {i: str(i) for i in self.orders_indices}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=16, font_weight='bold')
        
        handles = [plt.Line2D([0], [0], color=colors[idx], lw=2, label=f'Коммивояжер {idx+1}') 
                   for idx in range(used_salesmans)]
        handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Склад'))
        plt.legend(handles=handles)
        plt.title("Маршруты коммивояжеров")
        plt.savefig(output_file)

# Пример использования
if __name__ == "__main__":
    file_path = "input_data.txt"
    optimizer = SalesmanOptimization.from_file(file_path)
    used_salesmans, best_routes = optimizer.find_min_salesmans()
    if used_salesmans is not None:
        optimizer.plot_routes(used_salesmans, best_routes, 'routes.png')
