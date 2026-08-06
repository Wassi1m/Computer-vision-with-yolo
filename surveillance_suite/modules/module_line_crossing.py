"""
Module 3 : Franchissement de ligne virtuelle.

Principe :
- On suit chaque objet (track_id) via YOLO26 en mode tracking.
- Pour chaque track_id, on garde la position du centroïde à t-1 et t.
- On teste si le segment [position(t-1), position(t)] croise la ligne virtuelle.
- Test géométrique classique : orientation de 4 points (produit vectoriel).
"""


def _ccw(a, b, c):
    """True si a, b, c sont orientés dans le sens anti-horaire."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4):
    """True si le segment [p1,p2] croise le segment [p3,p4]."""
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


class LineCrossingCounter:
    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end
        self.last_positions = {}   # track_id -> (cx, cy)
        self.count_in = 0
        self.count_out = 0
        self.crossed_ids = set()   # évite de compter 2x la même frame

    def update(self, tracked_objects):
        """
        tracked_objects : liste de dicts {track_id, box(x1,y1,x2,y2), label}
        Retourne la liste des track_id qui viennent de franchir la ligne.
        """
        crossed_now = []
        for obj in tracked_objects:
            tid = obj["track_id"]
            if tid == -1:
                continue
            x1, y1, x2, y2 = obj["box"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            curr_pos = (cx, cy)

            if tid in self.last_positions:
                prev_pos = self.last_positions[tid]
                if segments_intersect(prev_pos, curr_pos, self.line_start, self.line_end):
                    # Sens du franchissement via le signe du produit vectoriel
                    dx_line = self.line_end[0] - self.line_start[0]
                    dy_line = self.line_end[1] - self.line_start[1]
                    dx_move = curr_pos[0] - prev_pos[0]
                    dy_move = curr_pos[1] - prev_pos[1]
                    cross = dx_line * dy_move - dy_line * dx_move

                    if cross > 0:
                        self.count_in += 1
                    else:
                        self.count_out += 1

                    crossed_now.append({"track_id": tid, "label": obj["label"]})

            self.last_positions[tid] = curr_pos

        return crossed_now
