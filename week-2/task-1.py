import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_to(self, other_point): # 畢氏定理
        return math.sqrt((self.x - other_point.x)**2 + (self.y - other_point.y)**2)

class Line:
    def __init__(self, point1, point2):
        self.p1 = Point(point1[0], point1[1])
        self.p2 = Point(point2[0], point2[1])
        
        # 計算線的斜率和y截距
        self.slope = (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x) if self.p2.x - self.p1.x != 0 else float('inf')
        self.b = self.p1.y - self.slope * self.p1.x if self.slope != float('inf') else None
    
    def is_parallel(self, other_line):
        # 兩條線的斜率相等即為平行
        if self.slope == float('inf') and other_line.slope == float('inf'):
            return True
        return abs(self.slope - other_line.slope) < 0.0001
    
    def is_perpendicular(self, other_line):
        # 兩條線的斜率相乘為-1即為垂直
        if self.slope == float('inf'):
            return other_line.slope == 0
        if other_line.slope == float('inf'):
            return self.slope == 0
        return abs(self.slope * other_line.slope + 1) < 0.0001

class Circle:
    def __init__(self, center, radius):
        self.center = Point(center[0], center[1])
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def intersects(self, other_circle):
        # 計算圓心距離
        center_distance = self.center.distance_to(other_circle.center)
        # 如果圓心距離小於半徑之和，則相交
        return center_distance < (self.radius + other_circle.radius)

class Polygon:
    def __init__(self, points):
        self.points = [Point(x, y) for x, y in points]
    def perimeter(self):
        total = 0
        n = len(self.points)
        for i in range(n):
            # 計算相鄰兩點之間的距離
            point1 = self.points[i]
            point2 = self.points[(i + 1) % n]  # 使用模運算處理最後一點到第一點的距離
            total += point1.distance_to(point2)
        return total

# 創建物件實例
line_a = Line((2, 4), (-6, 1))
line_b = Line((2, 2), (-6, -1))
line_c = Line((-1, 6), (-4, -4))

circle_a = Circle((6, 3), 2)
circle_b = Circle((8, 1), 1)

polygon_a = Polygon([(2, 0), (5, -1), (4, -4), (-1, -2)])

# 執行計算並輸出結果
print("Are Line A and Line B parallel? ", line_a.is_parallel(line_b)) # Line A 和 Line B 是否平行？
print("Are Line C and Line A perpendicular? ", line_c.is_perpendicular(line_a)) # Line C 和 Line A 是否垂直？
print("The area of Circle A: ", circle_a.area()) # Circle A 的面積
print("Do Circle A and Circle B intersect? ", circle_a.intersects(circle_b)) # Circle A 和 Circle B 是否相交
print("The perimeter of Polygon A: ", polygon_a.perimeter()) # Polygon A 的周長