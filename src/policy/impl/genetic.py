from ..policy import Policy
import numpy as np
from typing import List, Tuple
import random

class CuttingPattern:
    def __init__(self, stock_size: Tuple[int, int], items: List[Tuple[int, int, int]]):
        self.stock_width, self.stock_height = stock_size
        self.items = items  # List of (width, height, quantity)
        self.placement = []  # List of (x, y, width, height, item_idx)
        self.fitness = 0.0

    def calculate_fitness(self):
        """Calculate fitness based on material utilization and overlap penalty"""
        used_area = sum(w * h for _, _, w, h, _ in self.placement)
        total_area = self.stock_width * self.stock_height
        overlap_penalty = self._calculate_overlap()
        self.fitness = (used_area / total_area) - (overlap_penalty * 0.5)
        return self.fitness

    def _calculate_overlap(self):
        """Calculate overlap between placed items"""
        overlap_area = 0
        for i in range(len(self.placement)):
            for j in range(i + 1, len(self.placement)):
                x1, y1, w1, h1, _ = self.placement[i]
                x2, y2, w2, h2, _ = self.placement[j]
                
                overlap_width = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_height = min(y1 + h1, y2 + h2) - max(y1, y2)
                
                if overlap_width > 0 and overlap_height > 0:
                    overlap_area += overlap_width * overlap_height
        return overlap_area

class GeneticAlgorithm(Policy):
    def __init__(self, 
                 population_size=4,
                 generations=3,
                 mutation_rate=0.1,
                 elite_size=2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_solution = None

    def _initialize_population(self, stock_size, items):
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            pattern = CuttingPattern(stock_size, items)
            self._greedy_placement(pattern)
            population.append(pattern)
        return population

    def _greedy_placement(self, pattern):
        """Generate greedy placement using different strategies"""
        strategy = random.choice(['bottom_left', 'stack_height', 'stack_width', 'corner', 'skyline'])
        
        if strategy == 'bottom_left':
            self._bottom_left_placement(pattern)
        elif strategy == 'stack_height':
            self._stack_by_height(pattern)
        elif strategy == 'stack_width':
            self._stack_by_width(pattern)
        elif strategy == 'corner':
            self._corner_placement(pattern)
        else:
            self._skyline_placement(pattern)

    def _bottom_left_placement(self, pattern):
        """Place items at bottom-left most position"""
        for item_idx, (width, height, qty) in enumerate(pattern.items):
            for _ in range(qty):
                best_x, best_y = 0, pattern.stock_height
                
                # Try each position from bottom to top
                for y in range(pattern.stock_height - height, -1, -1):
                    for x in range(pattern.stock_width - width + 1):
                        can_place = True
                        
                        # Check if position is occupied
                        for px, py, pw, ph, _ in pattern.placement:
                            if (x < px + pw and x + width > px and
                                y < py + ph and y + height > py):
                                can_place = False
                                break
                        
                        if can_place and y >= best_y:
                            best_x, best_y = x, y
                            
                pattern.placement.append((best_x, best_y, width, height, item_idx))

    def _stack_by_height(self, pattern):
        """Stack items vertically by height"""
        current_x = 0
        items = [(width, height, qty, idx) for idx, (width, height, qty) 
                in enumerate(pattern.items)]
        items.sort(key=lambda x: (-x[1], -x[0]))  # Sort by height, then width
        
        for width, height, qty, item_idx in items:
            for _ in range(qty):
                if current_x + width > pattern.stock_width:
                    current_x = 0
                y = 0
                pattern.placement.append((current_x, y, width, height, item_idx))
                current_x += width

    def _stack_by_width(self, pattern):
        """Stack items horizontally by width"""
        current_y = 0
        items = [(width, height, qty, idx) for idx, (width, height, qty) 
                in enumerate(pattern.items)]
        items.sort(key=lambda x: (-x[0], -x[1]))  # Sort by width, then height
        
        for width, height, qty, item_idx in items:
            for _ in range(qty):
                if current_y + height > pattern.stock_height:
                    current_y = 0
                x = 0
                pattern.placement.append((x, current_y, width, height, item_idx))
                current_y += height

    def _corner_placement(self, pattern):
        """Place items in corners first"""
        corners = [(0, 0), (pattern.stock_width, 0),
                (0, pattern.stock_height), (pattern.stock_width, pattern.stock_height)]
        current_corner = 0
        
        items = [(width, height, qty, idx) for idx, (width, height, qty) 
                in enumerate(pattern.items)]
        items.sort(key=lambda x: -(x[0] * x[1]))  # Sort by area
        
        for width, height, qty, item_idx in items:
            for _ in range(qty):
                x, y = corners[current_corner]
                if current_corner in [1, 3]:
                    x -= width
                if current_corner in [2, 3]:
                    y -= height
                    
                pattern.placement.append((x, y, width, height, item_idx))
                current_corner = (current_corner + 1) % 4

    def _skyline_placement(self, pattern):
        """Place items using skyline strategy"""
        skyline = [0] * pattern.stock_width
        
        items = [(width, height, qty, idx) for idx, (width, height, qty) 
                in enumerate(pattern.items)]
        items.sort(key=lambda x: (-x[1], -x[0]))  # Sort by height, then width
        
        for width, height, qty, item_idx in items:
            for _ in range(qty):
                best_x, min_height = 0, float('inf')
                
                # Find position with minimum skyline
                for x in range(pattern.stock_width - width + 1):
                    max_height = max(skyline[x:x+width])
                    if max_height < min_height:
                        min_height = max_height
                        best_x = x
                
                # Update skyline
                for x in range(best_x, best_x + width):
                    skyline[x] = min_height + height
                    
                pattern.placement.append((best_x, min_height, width, height, item_idx))

    def _random_placement(self, pattern):
        """Choose random placement strategy for items"""
        # List of available placement strategies
        strategies = {
            'bottom_left': self._bottom_left_placement,
            'stack_height': self._stack_by_height,
            'stack_width': self._stack_by_width,
            'corner': self._corner_placement,
            'skyline': self._skyline_placement
        }
        
        # Randomly select a strategy
        strategy_name = random.choice(list(strategies.keys()))
        strategy_func = strategies[strategy_name]
        
        # Apply selected strategy
        strategy_func(pattern)
        
        # Add some randomization to the result
        if random.random() < 0.2:  # 20% chance to add mutation
            for _ in range(len(pattern.placement) // 3):  # Mutate 1/3 of placements
                if pattern.placement:
                    idx = random.randint(0, len(pattern.placement) - 1)
                    x, y, w, h, item_idx = pattern.placement[idx]
                    new_x = random.randint(0, pattern.stock_width - w)
                    new_y = random.randint(0, pattern.stock_height - h)
                    pattern.placement[idx] = (new_x, new_y, w, h, item_idx)
        

    def _selection(self, population):
        """Tournament selection with diversity preservation"""
        tournament_size = 3
        selected = []
        used_solutions = set()  # Track used solutions
        
        while len(selected) < self.population_size:
            # Select tournament candidates, excluding already used solutions
            available = [p for p in population if tuple(str(p.placement)) not in used_solutions]
            if not available:
                break  # If no unique solutions left, stop selection
                
            tournament = random.sample(available, min(tournament_size, len(available)))
            winner = max(tournament, key=lambda x: x.fitness)
            
            # Track this solution
            solution_key = tuple(str(winner.placement))
            if solution_key not in used_solutions:
                selected.append(winner)
                used_solutions.add(solution_key)
                
            # If we can't find enough unique solutions, fill with modified copies
            if len(available) < tournament_size:
                # Create modified copy of best solution
                best_copy = CuttingPattern(
                    (winner.stock_width, winner.stock_height),
                    winner.items
                )
                best_copy.placement = winner.placement.copy()
                self._mutation(best_copy)  # Force mutation
                selected.append(best_copy)
        
        return selected

    def _crossover(self, parent1, parent2):
        """Order crossover for placement sequences"""
        child = CuttingPattern((parent1.stock_width, parent1.stock_height), parent1.items)
        crossover_point = random.randint(0, len(parent1.placement))
        
        child.placement = parent1.placement[:crossover_point]
        remaining = [p for p in parent2.placement if p not in child.placement]
        child.placement.extend(remaining)
        
        return child

    def _mutation(self, pattern):
        """Enhanced mutation operator"""
        if random.random() < self.mutation_rate or not pattern.placement:
            mutations = random.randint(1, max(1, len(pattern.placement) // 3))
            for _ in range(mutations):
                mutation_type = random.choice(['shift', 'swap', 'rotate'])
                
                if mutation_type == 'shift' and pattern.placement:
                    # Shift item to new valid position
                    idx = random.randint(0, len(pattern.placement) - 1)
                    x, y, w, h, item_idx = pattern.placement[idx]
                    
                    # Try multiple positions
                    for _ in range(10):
                        new_x = random.randint(0, pattern.stock_width - w)
                        new_y = random.randint(0, pattern.stock_height - h)
                        
                        # Check if new position is valid
                        can_place = True
                        for i, (px, py, pw, ph, _) in enumerate(pattern.placement):
                            if i != idx and (new_x < px + pw and new_x + w > px and
                                        new_y < py + ph and new_y + h > py):
                                can_place = False
                                break
                        
                        if can_place:
                            pattern.placement[idx] = (new_x, new_y, w, h, item_idx)
                            break
                            
                elif mutation_type == 'swap' and len(pattern.placement) >= 2:
                    # Swap two items
                    idx1, idx2 = random.sample(range(len(pattern.placement)), 2)
                    pattern.placement[idx1], pattern.placement[idx2] = \
                        pattern.placement[idx2], pattern.placement[idx1]
                        
                elif mutation_type == 'rotate' and pattern.placement:
                    # Rotate item if possible
                    idx = random.randint(0, len(pattern.placement) - 1)
                    x, y, w, h, item_idx = pattern.placement[idx]
                    
                    if x + h <= pattern.stock_width and y + w <= pattern.stock_height:
                        pattern.placement[idx] = (x, y, h, w, item_idx)

    def get_action(self, observation, info):
        """Implement the genetic algorithm for cutting stock"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Convert products to required format
        items = [(p["size"][0], p["size"][1], p["quantity"]) 
                for p in products if p["quantity"] > 0]
        
        if not items:
            return {
                "stock_idx": -1,
                "size": np.array([0, 0]),
                "position": np.array([0, 0])
            }

        for stock_idx in range(len(stocks)):
            # Get stock dimensions
            stock = stocks[stock_idx]
            stock_size = self._get_stock_size_(stock)

            # Run genetic algorithm
            population = self._initialize_population(stock_size, items)
            
            for gen in range(self.generations):
                # Evaluate fitness
                for pattern in population:
                    pattern.calculate_fitness()
                
                # Sort by fitness
                population.sort(key=lambda x: x.fitness, reverse=True)
                
                # Store best solution
                if self.best_solution is None or population[0].fitness > self.best_solution.fitness:
                    self.best_solution = population[0]
                
                # Create new population
                new_population = population[:self.elite_size]
                
                # Selection
                selected = self._selection(population)
                
                # Crossover and mutation
                while len(new_population) < self.population_size:
                    parent1, parent2 = random.sample(selected, 2)
                    child = self._crossover(parent1, parent2)
                    self._mutation(child)
                    new_population.append(child)
                
                population = new_population

            # Convert best solution to action
            if self.best_solution and self.best_solution.placement:
                for placement in self.best_solution.placement:
                    x, y, w, h, item_idx = placement
                    if self._can_place_(stock, (x, y), (w, h)):
                        return {
                            "stock_idx": stock_idx,
                            "size": np.array([w, h]),
                            "position": np.array([x, y])
                        }
        
        return {
            "stock_idx": -1,
            "size": np.array([0, 0]),
            "position": np.array([0, 0])
        }

    def _calculate_waste(self, stock, x, y, w, h):
        """Calculate waste score for placing item"""
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Area utilization
        area_ratio = (w * h) / (stock_w * stock_h)
        
        # Edge alignment bonus
        edge_bonus = 0
        if x == 0 or x + w == stock_w:
            edge_bonus += 0.1
        if y == 0 or y + h == stock_h:
            edge_bonus += 0.1

        # Calculate waste score (lower is better)
        waste = (1 - area_ratio) - edge_bonus
        
        return waste