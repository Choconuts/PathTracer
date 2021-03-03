import cv2
import numpy as np
import random
import math
import glob
from functools import partial
import enum
import dataclasses
import logging


class Viewer:

    def __init__(self, path):
        self.path = path

    def show(self):
        glob.glob(f"{self.path}\\*.jpg")
        np.load()


arr = np.ndarray


def Vec(x=0., y=0., z=0.):
    return np.array([x, y, z])


def Ray(o, d):
    return np.row_stack([o, d])


def cross(a, b): return Vec(a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def norm(a): return a / np.linalg.norm(a)


DIFF = 'diff'
SPEC = 'spec'
REFR = 'refr'


@dataclasses.dataclass
class Sphere:
    rad: float
    p: arr
    e: arr
    c: arr
    refl: str

    def intersect(self, r: arr) -> float:
        op = self.p - r[0]
        b = np.dot(op, r[1])
        det = b * b - op.dot(op) + self.rad ** 2
        if det < 0:
            return 0
        else:
            det = math.sqrt(det)
        eps = 1e-4
        t = b - det
        if t > eps: return t
        t = b + det
        if t > eps: return t
        return 0


spheres = [
    Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF),
    Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF),
    Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF),
    Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF),
    Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),
    Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF),
    Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999, SPEC),
    Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * .999, REFR),
    Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFF)
]


spheres = [
    Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.55, .25, .25), DIFF),
    Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF),
    Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF),
    Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF),
    Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),
    Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF),
    Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999, SPEC),
    Sphere(16.5, Vec(73, 16.5, 78), Vec(2, 2, 0), Vec(1, 1, 1) * .999, REFR),
    Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(14, 8, 10), Vec(), DIFF)
]


def clamp(x): return 0 if x < 0 else 1 if x > 1 else x


def toInt(x): return int(pow(clamp(x), 1 / 2.2) * 255 + .5)


def intersect(r: arr, id: int):
    n = len(spheres)
    inf = t = 1e20
    for i in reversed(range(n)):
        d = spheres[i].intersect(r)
        if d > 0 and d < t:
            t = d
            id = i
    return t < inf, t, id


def erand(Xi): return random.random()


def radiance(r: arr, depth: int, Xi):
    id = 0
    it, t, id = intersect(r, id)
    if not it: return Vec()     # if miss, return black
    obj = spheres[id]        # hit obj
    x = r[0] + r[1] * t
    n = norm(x - obj.p)
    nl = n if np.dot(n, r[1]) < 0 else -n
    f = obj.c
    p = max(f)     # max refl
    depth += 1
    if depth > 5:
        if erand(Xi) < p:
            f = f / p
        else: return obj.e
    
    # max recur
    if depth > 10: return obj.e

    if obj.refl == DIFF:
        r1 = 2 * math.pi * erand(Xi)
        r2 = erand(Xi)
        r2s = np.sqrt(r2)
        w = nl
        u = Vec(0, 1) if math.fabs(w[0]) > .1 else Vec(1)
        u = cross(u, w)
        u = norm(u)
        v = cross(w, u)

        d = norm(u * math.cos(r1) * r2s + v * math.sin(r1) * r2s + w * math.sqrt(1 - r2))
        return obj.e + f * radiance(Ray(x, d), depth, Xi)

    elif obj.refl == SPEC:
        return obj.e + f * radiance(Ray(x, r[1] - n * 2 * np.dot(n, r[1])), depth, Xi)

    elif obj.refl == REFR:
        reflRay = Ray(x, r[1] - n * 2 * np.dot(n, r[1]))
        into = np.dot(n, nl) > 0
        nc = 1
        nt = 1.5
        nnt = nc/nt if into else nt/nc
        ddn = np.dot(r[1], nl)
        cos2t = 1 - nnt * nnt * (1 - ddn * ddn)
        if cos2t < 0:
            return obj.e + f * radiance(reflRay, depth, Xi)
        tdir = norm(r[1] * nnt - n * (1 if into else -1) * (ddn * nnt + np.sqrt(cos2t)))
        a = nt - nc
        b = nt + nc
        R0 = a * a / (b * b)
        c = 1 - (-ddn if into else np.dot(tdir, n))
        Re = R0 + (1 - R0) * c ** 5
        Tr = 1 - Re
        P = .25 + .5 * Re
        RP = Re / P
        TP = Tr / (1 - P)

        if depth > 2:
            if erand(Xi) < P:
                res = radiance(reflRay, depth, Xi) * RP
            else:
                res = radiance(Ray(x, tdir), depth, Xi) * TP
        else:
            res = radiance(reflRay, depth, Xi) * Re + radiance(Ray(x, tdir), depth, Xi) * Tr
        return obj.e + f * res


def radiance(r: arr, depth: int, Xi):
    # iterate version
    es = []
    fs = []

    for depth in range(12):
        it, t, id = intersect(r, 0)
        if not it:
            break
        obj = spheres[id]  # hit obj
        x = r[0] + r[1] * t
        n = norm(x - obj.p)
        nl = n if np.dot(n, r[1]) < 0 else -n
        f = obj.c
        p = max(f)  # max refl

        if depth > 5:
            if erand(Xi) < p:
                f = f / p
            else:
                es.append(obj.e)
                break

        # max recur
        if depth == 11:
            es.append(obj.e)
            break

        if obj.refl == DIFF:
            r1 = 2 * math.pi * erand(Xi)
            r2 = erand(Xi)
            r2s = np.sqrt(r2)
            w = nl
            u = Vec(0, 1) if math.fabs(w[0]) > .1 else Vec(1)
            u = cross(u, w)
            u = norm(u)
            v = cross(w, u)

            d = norm(u * math.cos(r1) * r2s + v * math.sin(r1) * r2s + w * math.sqrt(1 - r2))

            es.append(obj.e)
            fs.append(f)
            r = Ray(x, d)
            continue

        elif obj.refl == SPEC:
            es.append(obj.e)
            fs.append(f)
            r = Ray(x, r[1] - n * 2 * np.dot(n, r[1]))
            continue

        elif obj.refl == REFR:
            reflRay = Ray(x, r[1] - n * 2 * np.dot(n, r[1]))
            into = np.dot(n, nl) > 0
            nc = 1
            nt = 1.5
            nnt = nc / nt if into else nt / nc
            ddn = np.dot(r[1], nl)
            cos2t = 1 - nnt * nnt * (1 - ddn * ddn)
            if cos2t < 0:
                es.append(obj.e)
                fs.append(f)
                r = reflRay
                continue
            tdir = norm(r[1] * nnt - n * (1 if into else -1) * (ddn * nnt + np.sqrt(cos2t)))
            a = nt - nc
            b = nt + nc
            R0 = a * a / (b * b)
            c = 1 - (-ddn if into else np.dot(tdir, n))
            Re = R0 + (1 - R0) * c ** 5
            Tr = 1 - Re
            P = .25 + .5 * Re
            RP = Re / P
            TP = Tr / (1 - P)

            if erand(Xi) < P:
                f *= RP
                r = reflRay
            else:
                f *= TP
                r = Ray(x, tdir)

            es.append(obj.e)
            fs.append(f)
            continue

    assert len(es) - len(fs) == 1
    res = [0, 0, 0]
    for i, e in reversed(list(enumerate(es))):
        if i > 0:
            for ch in range(3):
                res[ch] += e[ch]
                res[ch] *= fs[i - 1][ch]

    for ch in range(3):
        res[ch] += es[0][ch]
    return np.array(res)

def main():
    w, h = 200, 150
    samps = 10
    cam = Ray(Vec(50, 52, 295.6), norm(Vec(0, -0.042612, -1)))
    cx = Vec(w * .5135 / h)
    cy = norm(cross(cx, cam[1])) * .5135
    c = np.zeros([w * h, 3])
    for y in range(h):          # Loop over image rows
        logging.error(f"\rRendering ({samps*4} spp) {100.*y/(h-1)}%")
        Xi = [0, 0, y*y*y]
        for x in range(w):
            for sy in range(2):     # 2x2 subpixel rows
                i = (h - y - 1) * w + x
                for sx in range(2):
                    r = Vec()
                    for s in range(samps):
                        r1 = 2 * erand(Xi)
                        dx = math.sqrt(r1) - 1 if r1 < 1 else 1 - math.sqrt(2-r1)
                        r2 = 2 * erand(Xi)
                        dy = math.sqrt(r2) - 1 if r2 < 1 else 1 - math.sqrt(2 - r2)
                        d = cx * (((sx+.5 + dx)/2 + x)/w - .5) + \
                            cy * (((sy+.5 + dy)/2 + y)/h - .5) + cam[1]

                        r = r + radiance(Ray(cam[0] + d * 140, norm(d)), 0, Xi) * (1./samps)
                    c[i] = c[i] + Vec(clamp(r[0]), clamp(r[1]), clamp(r[2])) * .25

    for v in c:
        v[0] = toInt(v[0])
        v[1] = toInt(v[1])
        v[2] = toInt(v[2])

    cv2.imwrite('out_007.jpg', c.reshape([h, w, 3]))
    cv2.imshow('out', c.reshape([h, w, 3]))


if __name__ == '__main__':
    main()
