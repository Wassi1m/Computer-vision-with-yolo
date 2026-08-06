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
    # Nombre de frames consécutives sans détection avant d'oublier un track_id
    # (évite une fuite mémoire sur les flux longue durée : sans ça,
    # last_positions/crossed_ids grossissent indéfiniment avec chaque nouveau
    # track_id jamais revu, ex: caméra 24/7 avec des centaines de passages/jour).
    MAX_MISSING_FRAMES = 30
    # Distance perpendiculaire (px) que le centroïde doit parcourir depuis la
    # ligne avant qu'un même track_id puisse compter un nouveau franchissement.
    # Sans ce cooldown, un léger jitter du centroïde pile sur la ligne peut
    # faire osciller le point de part et d'autre et compter plusieurs
    # franchissements pour le même passage réel.
    RECROSS_MARGIN = 20

    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end
        self.last_positions = {}   # track_id -> (cx, cy)
        self.count_in = 0
        self.count_out = 0
        self.crossed_ids = set()   # track_id en cooldown depuis leur dernier franchissement
        self._missing_frames = {}  # track_id -> nb de frames consécutives sans détection

    def _line_distance(self, pos):
        """Distance perpendiculaire (non signée) d'un point à la ligne."""
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        dx, dy = x2 - x1, y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length < 1e-6:
            return 0.0
        return abs(dx * (y1 - pos[1]) - (x1 - pos[0]) * dy) / length

    def _prune_stale_tracks(self, seen_ids):
        for tid in list(self.last_positions):
            if tid in seen_ids:
                self._missing_frames[tid] = 0
                continue
            self._missing_frames[tid] = self._missing_frames.get(tid, 0) + 1
            if self._missing_frames[tid] > self.MAX_MISSING_FRAMES:
                self.last_positions.pop(tid, None)
                self.crossed_ids.discard(tid)
                self._missing_frames.pop(tid, None)

    def update(self, tracked_objects):
        """
        tracked_objects : liste de dicts {track_id, box(x1,y1,x2,y2), label}
        Retourne la liste des track_id qui viennent de franchir la ligne.
        """
        crossed_now = []
        seen_ids = set()
        for obj in tracked_objects:
            tid = obj["track_id"]
            if tid == -1:
                continue
            seen_ids.add(tid)
            x1, y1, x2, y2 = obj["box"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            curr_pos = (cx, cy)

            if tid in self.crossed_ids and self._line_distance(curr_pos) > self.RECROSS_MARGIN:
                self.crossed_ids.discard(tid)

            if tid in self.last_positions:
                prev_pos = self.last_positions[tid]
                if tid not in self.crossed_ids and segments_intersect(
                    prev_pos, curr_pos, self.line_start, self.line_end
                ):
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
                    self.crossed_ids.add(tid)

            self.last_positions[tid] = curr_pos

        self._prune_stale_tracks(seen_ids)
        return crossed_now
