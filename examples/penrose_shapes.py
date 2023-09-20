# based on https://rosettacode.org/wiki/Penrose_tiling

import numpy as np
import napari

class Tile:
    def __init__(self, t, x, y, a, s):
        self.type = t
        self.x = x
        self.y = y
        self.angle = a
        self.size = s

    def __eq__(self, other):
        return (self.type == other.type and self.x == other.x and self.y == other.y and self.angle == other.angle)

    def __hash__(self):
        return hash((self.type, self.x, self.y, self.angle))

class Type:
    Kite = 1
    Dart = 2

class PenroseTiling:

    G = (1 + np.sqrt(5)) / 2  # golden ratio
    T = np.radians(36)  # theta

    def __init__(self, num_generations=5):
        self.tiles = []
        w, h = 700, 450
        self.tiles = self.deflateTiles(self.setupPrototiles(w, h), num_generations)

    def setupPrototiles(self, w, h):
        proto = []

        # sun
        for a in np.arange(np.pi / 2 + self.T, 3 * np.pi, 2 * self.T):
            proto.append(Tile(Type.Kite, w / 2, h / 2, a, w / 2.5))

        return proto

    def deflateTiles(self, tls, generation):
        if generation <= 0:
            return tls

        next_tiles = []

        for tile in tls:
            x, y, a = tile.x, tile.y, tile.angle
            nx, ny = None, None
            size = tile.size / self.G

            if tile.type == Type.Dart:
                next_tiles.append(Tile(Type.Kite, x, y, a + 5 * self.T, size))
                for i, sign in [(0, 1), (1, -1)]:
                    nx = x + np.cos(a - 4 * self.T * sign) * self.G * tile.size
                    ny = y - np.sin(a - 4 * self.T * sign) * self.G * tile.size
                    next_tiles.append(Tile(Type.Dart, nx, ny, a - 4 * self.T * sign, size))

            else:
                for i, sign in [(0, 1), (1, -1)]:
                    next_tiles.append(Tile(Type.Dart, x, y, a - 4 * self.T * sign, size))
                    nx = x + np.cos(a - self.T * sign) * self.G * tile.size
                    ny = y - np.sin(a - self.T * sign) * self.G * tile.size
                    next_tiles.append(Tile(Type.Kite, nx, ny, a + 3 * self.T * sign, size))

        tls = list(set(next_tiles))

        return self.deflateTiles(tls, generation - 1)

    def drawTiles(self):
        polygons = []

        dist = [[self.G, self.G, self.G], [-self.G, -1, -self.G]]

        for tile in self.tiles:
            angle = tile.angle - self.T
            coords = [(tile.x, tile.y)]
            
            ord_val = 0 if tile.type == Type.Kite else 1
            for i in range(3):
                x = tile.x + dist[ord_val][i] * tile.size * np.cos(angle)
                y = tile.y - dist[ord_val][i] * tile.size * np.sin(angle)
                coords.append((x, y))
                angle += self.T

            coords.append((tile.x, tile.y))
            polygons.append(coords)

        return polygons

if __name__ == "__main__":
    tiling = PenroseTiling(num_generations=9)
    polygons = tiling.drawTiles()
    print(f"Number of polygons: {len(polygons)}")

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_shapes(polygons, shape_type='polygon', face_color=['orange' if len(polygon) == 5 else 'yellow' for polygon in polygons], edge_color='darkgray')
