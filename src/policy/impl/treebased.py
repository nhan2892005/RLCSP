from ..policy import Policy
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Node:
    x: int  # x coordinate
    y: int  # y coordinate
    width: int  # width of space
    height: int  # height of space
    used: bool = False
    right: Optional['Node'] = None
    bottom: Optional['Node'] = None

class TreeBasedHeuristic(Policy):
    def __init__(self):
        self.trees = {}  # Dictionary to store trees for each stock
        self.current_stock = 0
        self.min_waste_threshold = 0.1

    def _create_node(self, x: int, y: int, width: int, height: int) -> Node:
        """Create a new tree node representing a rectangular space"""
        return Node(x=x, y=y, width=width, height=height)

    def _find_node(self, root: Node, width: int, height: int) -> Optional[Node]:
        """Find suitable node that fits the product dimensions using best-fit strategy"""
        if root.used:
            # Try right branch
            right_node = self._find_node(root.right, width, height) if root.right else None
            if right_node:
                return right_node
                
            # Try bottom branch
            return self._find_node(root.bottom, width, height) if root.bottom else None

        # Check if product fits in current node
        if width <= root.width and height <= root.height:
            # Check if node is perfect fit
            if width == root.width and height == root.height:
                return root
                
            # Return current node as it can accommodate the product
            return root
            
        return None

    def _split_node(self, node: Node, width: int, height: int) -> bool:
        """Split node into two parts after placing a product"""
        if node.used:
            return False

        # Calculate remaining space
        remaining_right = node.width - width
        remaining_bottom = node.height - height

        # Split horizontally if wider, vertically if taller
        if remaining_right > remaining_bottom:
            # Create right node
            node.right = self._create_node(
                node.x + width,
                node.y,
                remaining_right,
                height
            )
            # Create bottom node
            node.bottom = self._create_node(
                node.x,
                node.y + height,
                node.width,
                remaining_bottom
            )
        else:
            # Create bottom node
            node.bottom = self._create_node(
                node.x,
                node.y + height,
                width,
                remaining_bottom
            )
            # Create right node
            node.right = self._create_node(
                node.x + width,
                node.y,
                remaining_right,
                node.height
            )

        node.used = True
        return True

    def _calculate_utilization(self, node: Node, product_area: int) -> float:
        """Calculate space utilization for placing product in node"""
        if not node:
            return 0.0
            
        node_area = node.width * node.height
        if node_area == 0:
            return 0.0
            
        return product_area / node_area

    def get_action(self, observation, info):
        """Get next action using tree-based placement strategy"""
        stocks = observation["stocks"]
        products = [p for p in observation["products"] if p["quantity"] > 0]
        
        if not products:
            return {
                "stock_idx": -1,
                "size": np.array([0, 0]),
                "position": np.array([0, 0])
            }

        # Sort products by area (largest first)
        products.sort(key=lambda x: (
            -(x['size'][0] * x['size'][1]),  # Area
            -max(x['size']),  # Longest side
            -min(x['size'])   # Shortest side
        ))

        best_action = None
        best_utilization = -1

        # Try each product
        for prod in products:
            if prod["quantity"] <= 0:
                continue

            prod_w, prod_h = prod["size"]
            prod_area = prod_w * prod_h

            # Try each stock
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                
                # Initialize tree for stock if not exists
                if stock_idx not in self.trees:
                    self.trees[stock_idx] = self._create_node(0, 0, stock_w, stock_h)

                # Try both orientations
                for w, h in [(prod_w, prod_h), (prod_h, prod_w)]:
                    if w > stock_w or h > stock_h:
                        continue

                    # Find suitable node
                    node = self._find_node(self.trees[stock_idx], w, h)
                    if node:
                        # Calculate utilization
                        utilization = self._calculate_utilization(node, prod_area)
                        
                        if utilization > best_utilization:
                            best_utilization = utilization
                            best_action = {
                                "stock_idx": stock_idx,
                                "size": np.array([w, h]),
                                "position": np.array([node.x, node.y])
                            }

                            # Split node if utilization is good enough
                            if utilization > self.min_waste_threshold:
                                self._split_node(node, w, h)
                                return best_action

        if best_action:
            # Split the chosen node
            node = self._find_node(
                self.trees[best_action["stock_idx"]], 
                best_action["size"][0],
                best_action["size"][1]
            )
            self._split_node(
                node,
                best_action["size"][0],
                best_action["size"][1]
            )
            return best_action

        return {
            "stock_idx": -1,
            "size": np.array([0, 0]),
            "position": np.array([0, 0])
        }

    def _validate_placement(self, stock, position, size) -> bool:
        """Validate if placement is possible"""
        x, y = position
        w, h = size
        
        # Check boundaries
        stock_w, stock_h = self._get_stock_size_(stock)
        if x + w > stock_w or y + h > stock_h:
            return False
            
        # Check if space is empty
        return self._can_place_(stock, position, size)