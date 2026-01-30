
import cv2
import numpy as np
import os
from pathlib import Path
from skimage.morphology import skeletonize
from collections import deque


class PipetteAngleDetector:

    
    def __init__(self, tip_portion_ratio=0.3):

        self.tip_portion_ratio = tip_portion_ratio
    
    def load_yolo_polygon(self, txt_path, img_width, img_height):
        
        polygons = []
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                # Convert normalized coordinates to pixel coordinates
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    points.append([x, y])
                
                polygons.append({
                    'class_id': class_id,
                    'points': np.array(points, dtype=np.int32)
                })
        
        return polygons
    
    def create_mask(self, polygon_points, img_height, img_width):
        """Create binary mask from polygon points."""
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)
        return mask
    
    def compute_skeleton(self, mask):
      
        skeleton = skeletonize(mask > 0).astype(np.uint8) * 255
        return skeleton
    
    def find_skeleton_endpoints(self, skeleton):
      
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = (neighbor_count == 1) & (skeleton > 0)
        coords = np.argwhere(endpoints)
        
        return [(int(c[1]), int(c[0])) for c in coords]  # Convert to (x, y)
    
    def find_farthest_endpoints(self, skeleton):
       
        endpoints = self.find_skeleton_endpoints(skeleton)
        
        if len(endpoints) < 2:
            # Fallback: use any skeleton points
            points = np.argwhere(skeleton > 0)
            if len(points) < 2:
                return None, None
            endpoints = [(int(p[1]), int(p[0])) for p in points]
        
        max_dist = 0
        best_pair = (endpoints[0], endpoints[-1])
        
        for i, p1 in enumerate(endpoints):
            for p2 in endpoints[i + 1:]:
                dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (p1, p2)
        
        return best_pair
    
    def trace_skeleton_path(self, skeleton, start, end):
        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        parent = {}
        
        queue = deque([start])
        visited[start[1], start[0]] = True
        parent[start] = None
        
        # 8-connected neighborhood
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while queue:
            current = queue.popleft()
            
            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if skeleton[ny, nx] > 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        parent[(nx, ny)] = current
                        queue.append((nx, ny))
        
        return []
    
    def measure_width_at_point(self, mask, point, direction):

        h, w = mask.shape
        px, py = int(point[0]), int(point[1])
        
        # Perpendicular direction
        perp_x, perp_y = -direction[1], direction[0]
        length = np.hypot(perp_x, perp_y)
        if length > 0:
            perp_x, perp_y = perp_x / length, perp_y / length
        else:
            return 0
        
        width = 0
        for dist in range(-300, 301):
            sx = int(px + dist * perp_x)
            sy = int(py + dist * perp_y)
            if 0 <= sx < w and 0 <= sy < h:
                if mask[sy, sx] > 0:
                    width += 1
        
        return width
    
    def identify_tip_and_handle(self, mask, path):

        if len(path) < 10:
            return path[0], path[-1], path
        
        # Overall direction
        dx = path[-1][0] - path[0][0]
        dy = path[-1][1] - path[0][1]
        length = np.hypot(dx, dy)
        direction = (dx / length, dy / length) if length > 0 else (1, 0)
        
        # Sample widths at both ends (first and last 20%)
        n = len(path)
        first_region = path[:max(1, n // 5)]
        last_region = path[n - max(1, n // 5):]
        
        # Measure widths
        step = max(1, len(first_region) // 3)
        widths_first = [self.measure_width_at_point(mask, pt, direction) 
                       for pt in first_region[::step]]
        
        step = max(1, len(last_region) // 3)
        widths_last = [self.measure_width_at_point(mask, pt, direction) 
                      for pt in last_region[::step]]
        
        avg_first = np.mean(widths_first) if widths_first else 0
        avg_last = np.mean(widths_last) if widths_last else 0
        
        if avg_first > avg_last:
            # First is handle (wide), last is tip (narrow)
            return path[0], path[-1], path
        else:
            # Last is handle (wide), first is tip (narrow)
            return path[-1], path[0], path[::-1]
    
    def fit_line_direction(self, points):
    
        if len(points) < 2:
            return (1, 0)
        
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        vx, vy, _, _ = cv2.fitLine(points_array, cv2.DIST_L2, 0, 0.01, 0.01)
        
        return (float(vx), float(vy))
    
    def calculate_angle(self, direction):
   
        vx, vy = direction
        angle = np.degrees(np.arctan2(abs(vy), abs(vx)))
        return angle
    
    def detect(self, image_path, annotation_path):
      
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Load annotation
        polygons = self.load_yolo_polygon(annotation_path, w, h)
        if not polygons:
            return None
        
        polygon = polygons[0]
        points = polygon['points']
        
        # Create mask and skeleton
        mask = self.create_mask(points, h, w)
        skeleton = self.compute_skeleton(mask)
        
        # Find skeleton endpoints
        end1, end2 = self.find_farthest_endpoints(skeleton)
        if end1 is None or end2 is None:
            return None
        
        # Trace skeleton path
        skeleton_path = self.trace_skeleton_path(skeleton, end1, end2)
        if len(skeleton_path) < 2:
            return None
        
        # Identify tip and handle
        handle_point, tip_point, oriented_path = self.identify_tip_and_handle(mask, skeleton_path)
        
        # Calculate angle at tip portion
        n = len(oriented_path)
        tip_start = int(n * (1 - self.tip_portion_ratio))
        tip_portion = oriented_path[tip_start:]
        
        if len(tip_portion) < 5:
            tip_portion = oriented_path[n // 2:]
        
        tip_direction = self.fit_line_direction(tip_portion)
        angle = self.calculate_angle(tip_direction)
        
        return {
            'angle': angle,
            'handle_point': handle_point,
            'tip_point': tip_point,
            'skeleton_path': oriented_path,
            'polygon_points': points,
            'image_size': (w, h)
        }
    
    def visualize(self, image, result, output_path=None):
   
        output = image.copy()
        
        # Draw polygon outline (yellow)
        cv2.polylines(output, [result['polygon_points']], True, (0, 255, 255), 2)
        
        # Draw skeleton path (green) - follows centerline
        path = result['skeleton_path']
        step = max(1, len(path) // 100)
        simplified = path[::step]
        
        for i in range(len(simplified) - 1):
            cv2.line(output, simplified[i], simplified[i + 1], (0, 255, 0), 3)
        
        # Draw handle point P1 (green)
        p1 = result['handle_point']
        cv2.circle(output, p1, 12, (0, 255, 0), -1)
        cv2.putText(output, "P1 (wide)", (p1[0] + 15, p1[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw tip point P2 (red)
        p2 = result['tip_point']
        cv2.circle(output, p2, 12, (0, 0, 255), -1)
        cv2.putText(output, "P2 (tip)", (p2[0] + 15, p2[1] + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw horizontal reference line at tip
        cv2.line(output, (p2[0] - 200, p2[1]), (p2[0] + 200, p2[1]), (0, 0, 255), 2)
        
        # Display angle
        cv2.putText(output, f"Angle: {result['angle']:.1f} deg", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        if output_path:
            cv2.imwrite(output_path, output)
        
        return output


def process_dataset(image_folder, label_folder, output_folder, csv_path=None):

    os.makedirs(output_folder, exist_ok=True)
    
    detector = PipetteAngleDetector()
    extensions = {'.jpg', '.jpeg', '.png'}
    results = []
    
    for filename in sorted(os.listdir(image_folder)):
        if Path(filename).suffix.lower() not in extensions:
            continue
        
        image_path = os.path.join(image_folder, filename)
        txt_path = os.path.join(label_folder, Path(filename).stem + '.txt')
        output_path = os.path.join(output_folder, filename)
        
        if not os.path.exists(txt_path):
            print(f"No annotation for {filename}, skipping...")
            continue
        
        # Detect
        result = detector.detect(image_path, txt_path)
        
        if result is None:
            print(f"Detection failed for {filename}")
            continue
        
        # Visualize and save
        image = cv2.imread(image_path)
        detector.visualize(image, result, output_path)
        
        results.append((filename, result['angle']))
        print(f"{filename}: {result['angle']:.1f}°")
    
    # Save CSV
    if csv_path and results:
        with open(csv_path, 'w') as f:
            f.write("filename,angle\n")
            for filename, angle in results:
                f.write(f"{filename},{angle:.2f}\n")
        print(f"\nResults saved to {csv_path}")
    
    print(f"\nProcessed {len(results)} images")
    return results


if __name__ == "__main__":

    
    process_dataset(
        image_folder="input/images",       # Folder containing your images
        label_folder="input/labels",       # Folder containing .txt annotation files
        output_folder="./output",      # Folder to save annotated images
        csv_path="./results.csv"       # CSV file to save angles (or None to skip)
    )
    
