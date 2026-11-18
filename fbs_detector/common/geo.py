"""地理计算与空间聚类工具（仅依赖标准库）。

坐标采用经纬度（WGS-84 近似平面化处理：园区尺度上用等距圆柱投影转米），
对几公里范围内的园区定位精度足够，且避免外部 GIS 依赖。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_M = 6_371_000.0


def project_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """经纬度 -> 以参考点为原点的局部平面坐标（米）。"""
    x = math.radians(lon - ref_lon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    y = math.radians(lat - ref_lat) * EARTH_RADIUS_M
    return x, y


def inverse_xy(x: float, y: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """局部平面坐标（米）-> 经纬度。"""
    lat = ref_lat + math.degrees(y / EARTH_RADIUS_M)
    lon = ref_lon + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat))))
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点间大圆距离（米）。"""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass
class GeoEstimate:
    """发射源位置估计结果。"""
    lat: float
    lon: float
    radius_m: float          # 不确定半径（观测散布 + 估计残差）
    n_points: int            # 使用的观测点数
    centroid_spread_m: float # 观测点相对质心的中位距离


def weighted_centroid(points: list[tuple[float, float, float]]) -> GeoEstimate | None:
    """加权质心定位。

    points: [(lat, lon, weight), ...]，weight 一般取信号强度归一值（越近权重越大）。
    伪基站定位只能靠众包观测的信号地理分布反推，给不出精确点，因此同时输出
    不确定半径（观测散布 + 信号反演残差）。
    """
    if not points:
        return None
    ref_lat = sum(p[0] for p in points) / len(points)
    ref_lon = sum(p[1] for p in points) / len(points)
    xy = [(*project_xy(la, lo, ref_lat, ref_lon), w) for la, lo, w in points]
    wsum = sum(w for _, _, w in xy) or 1.0
    cx = sum(x * w for x, _, w in xy) / wsum
    cy = sum(y * w for _, y, w in xy) / wsum
    dists = sorted(math.hypot(x - cx, y - cy) for x, y, _ in xy)
    spread = dists[len(dists) // 2]  # 中位数，抗个别漂移点
    # 信号反演本身有百米级残差，与观测散布合并为不确定半径
    residual = 120.0
    radius = max(100.0, spread * 1.25 + residual)
    lat, lon = inverse_xy(cx, cy, ref_lat, ref_lon)
    return GeoEstimate(lat=lat, lon=lon, radius_m=radius,
                       n_points=len(points), centroid_spread_m=spread)


def dbscan_xy(points: list[tuple[float, float]], eps_m: float, min_samples: int) -> list[int]:
    """极简 DBSCAN（局部米制坐标），返回每个点的簇标签，噪声为 -1。

    用于把同一伪基站身份漂移产生的、彼此邻近的多个候选发射源聚成一个事件，
    避免一个伪基站被报成多起事件。
    """
    n = len(points)
    labels = [-99] * n  # -99=未访问, -1=噪声
    # 邻接表（园区点数小，直接 O(n^2)）
    neighbors: list[list[int]] = []
    for i, (x1, y1) in enumerate(points):
        nbr = [j for j, (x2, y2) in enumerate(points)
               if math.hypot(x1 - x2, y1 - y2) <= eps_m]
        neighbors.append(nbr)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -99:
            continue
        if len(neighbors[i]) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = list(neighbors[i])
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] == -99:
                labels[j] = cluster_id
                if len(neighbors[j]) >= min_samples:
                    for q in neighbors[j]:
                        if q not in seeds:
                            seeds.append(q)
            k += 1
        cluster_id += 1
    return [lab if lab != -99 else -1 for lab in labels]
