#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : ti_pt.py
@Author: Chen Yanzhen
@Date  : 2021/3/1 16:41
@Desc  : 
"""
import cv2
import numpy as np
import random
import glob
import taichi as ti
import enum
import dataclasses
import logging

ti.init()

WIDTH, HEIGHT = 200, 150
SAMPLES = 100
DEPTH = 10
PI = 3.1415927
colors = ti.Vector.field(3, ti.float32, shape=(WIDTH * HEIGHT, 2, 2, SAMPLES))
emissions = ti.Vector.field(3, ti.float32, shape=(WIDTH * HEIGHT * 4 * SAMPLES, DEPTH))
brdfs = ti.Vector.field(3, ti.float32, shape=(WIDTH * HEIGHT * 4 * SAMPLES, DEPTH))


radius = ti.field(ti.float32, shape=9)
positions = ti.Vector.field(3, ti.float32, shape=9)
sphere_emissions = ti.Vector.field(3, ti.float32, shape=9)
sphere_colors = ti.Vector.field(3, ti.float32, shape=9)
sphere_refls = ti.field(ti.int8, shape=9)


DIFF = 0
SPEC = 1
REFR = 2


@ti.func
def clamp(x):
    return 0 if x < 0 else 1 if x > 1 else x


@ti.func
def zero_vec():
    return ti.Vector([0., 0., 0.])


@ti.kernel
def init():
    for i in range(6):
        radius[i] = 1e5
    radius[6] = 16.5
    radius[7] = 16.5
    radius[8] = 600
    positions[0] = ti.Vector([1e5 + 1, 40.8, 81.6])
    positions[1] = ti.Vector([-1e5 + 99, 40.8, 81.6])
    positions[2] = ti.Vector([50, 40.8, 1e5])
    positions[3] = ti.Vector([50, 40.8, -1e5 + 170])
    positions[4] = ti.Vector([50, 1e5, 81.6])
    positions[5] = ti.Vector([50, -1e5 + 81.6, 81.6])
    positions[6] = ti.Vector([27, 16.5, 47])
    positions[7] = ti.Vector([73, 16.5, 78])
    positions[8] = ti.Vector([50, 681.6 - .27, 81.6])

    for i in range(8):
        sphere_emissions[i] = zero_vec()
    sphere_emissions[8] = ti.Vector([12, 12, 12.])

    sphere_colors[0] = ti.Vector([.75, .25, .25])
    sphere_colors[1] = ti.Vector([.25, .25, .75])
    sphere_colors[2] = ti.Vector([.75, .75, .75])
    sphere_colors[3] = ti.Vector([0., 0., 0.])
    sphere_colors[4] = ti.Vector([.75, .75, .75])
    sphere_colors[5] = ti.Vector([.75, .75, .75])
    sphere_colors[6] = ti.Vector([1, 1, 1]) * .999
    sphere_colors[7] = ti.Vector([1, 1, 1]) * .999
    sphere_colors[8] = ti.Vector([0., 0., 0.])

    for i in range(6):
        sphere_refls[i] = DIFF
    sphere_refls[6] = SPEC
    sphere_refls[7] = SPEC
    sphere_refls[8] = DIFF

    for I in ti.grouped(emissions):
        emissions[I] *= 0


@ti.func
def intersect_sphere(rp, rd, i):
    res = 0
    op = positions[i] - rp
    b = ti.dot(op, rd)
    det = b * b - op.dot(op) + radius[i] ** 2
    if det < 0:
        res = 0
    else:
        det = ti.sqrt(det)
        eps = 1e-4
        t = b - det
        if t > eps:
            res = t
        else:
            t = b + det
            if t > eps:
                res = t
    return res


@ti.func
def intersect(rp, rd):
    min_t = -1
    hid = 4
    for i in range(9):
        t = intersect_sphere(rp, rd, i)
        if min_t < 0 or 0 < t < min_t:
            min_t = t
            hid = i
    x = rp + rd * min_t
    normal = (x - positions[hid]).normalized()
    return min_t, sphere_colors[hid], normal, sphere_emissions[hid], sphere_refls[hid]
    # return 0.2, ti.Vector([0.3, 0.4, 0.5]), ti.Vector([0, 1., 0]), ti.Vector([0.8, 0.7, 0.6]), 0

# @ti.kernel
# def radiance(rp, rd, depth):
#     hit, t, f, n, e, refl = intersect(rp, rd)
#     x = rp + rd * t
#     if not hit:
#         return ti.Vector([0, 0, 0])
#     nl = n if np.dot(n, rd) < 0 else -n
#     p = max(f)  # max refl
#     depth += 1
#     if depth > 5:
#         if ti.random() < p:
#             f = f / p
#         else:
#             return e
#
#     # max recur
#     if depth > 10: return e
#
#     if refl == DIFF:
#         r1 = 2 * math.pi * ti.random()
#         r2 = ti.random()
#         r2s = np.sqrt(r2)
#         w = nl
#         u = ti.Vector([0, 1, 0]) if ti.abs(w[0]) > .1 else ti.Vector([1, 1, 1])
#         u = ti.cross(u, w)
#         u = ti.normalized(u)
#         v = ti.cross(w, u)
#
#         d = ti.normalized(u * math.cos(r1) * r2s + v * math.sin(r1) * r2s + w * math.sqrt(1 - r2))
#         return e + f * radiance(x, d, depth)
#
#     elif refl == SPEC:
#         return e + f * radiance(x, rd - n * 2 * np.dot(n, rd), depth)
#
#     elif refl == REFR:
#         # reflRay = x, r[1] - n * 2 * np.dot(n, r[1])
#         # into = np.dot(n, nl) > 0
#         # nc = 1
#         # nt = 1.5
#         # nnt = nc / nt if into else nt / nc
#         # ddn = np.dot(r[1], nl)
#         # cos2t = 1 - nnt * nnt * (1 - ddn * ddn)
#         # if cos2t < 0:
#         #     return e + f * radiance(reflRay, depth, Xi)
#         # tdir = ti.normalized(r[1] * nnt - n * (1 if into else -1) * (ddn * nnt + np.sqrt(cos2t)))
#         # a = nt - nc
#         # b = nt + nc
#         # R0 = a * a / (b * b)
#         # c = 1 - (-ddn if into else np.dot(tdir, n))
#         # Re = R0 + (1 - R0) * c ** 5
#         # Tr = 1 - Re
#         # P = .25 + .5 * Re
#         # RP = Re / P
#         # TP = Tr / (1 - P)
#         #
#         # if depth > 2:
#         #     if ti.random() < P:
#         #         res = radiance(reflRay, depth, Xi) * RP
#         #     else:
#         #         res = radiance(Ray(x, tdir), depth, Xi) * TP
#         # else:
#         #     res = radiance(reflRay, depth, Xi) * Re + radiance(Ray(x, tdir), depth, Xi) * Tr
#         # return e + f * res
#         return e


@ti.func
def radiance(rp, rd, sid):

    test = ti.Matrix(rows=[[0, 1, 2]] * 12)

    for depth in range(DEPTH):
        print(depth)
        t, f, n, e, refl = intersect(rp, rd)
        if t < 0:
            break
        x = rp + rd * t
        nl = n if n.dot(rd) < 0 else -n
        p = f.max()

        if depth > 5:
            if ti.random() < p:
                f = f / p
            else:
                emissions[sid, depth] = e
                brdfs[sid, depth] *= 0
                break

        # max recur
        if depth == DEPTH - 1:
            emissions[sid, depth] = e
            brdfs[sid, depth] *= 0
            break

        if refl == DIFF:
            r1 = 2 * PI * ti.random()
            r2 = ti.random()
            r2s = ti.sqrt(r2)
            w = nl
            u = ti.Vector([0., 1., 0.]) if ti.abs(w[0]) > .1 else ti.Vector([1, 0., 0])
            u = u.cross(w)
            u = u.normalized()
            v = w.cross(u)

            d = u * ti.cos(r1) * r2s + v * ti.sin(r1) * r2s + w * ti.sqrt(1 - r2)
            d = d.normalized()

            emissions[sid, depth] = e
            brdfs[sid, depth] = f
            rp, rd = x, d

        elif refl == SPEC:
            emissions[sid, depth] = e
            brdfs[sid, depth] = f
            rp, rd = x, rd - n * 2 * n.dot(rd)
            continue

        elif refl == REFR:
            reflRay = x, rd - n * 2 * n.dot(rd)
            into = n.dot(nl) > 0
            nc = 1
            nt = 1.5
            nnt = nc / nt if into else nt / nc
            ddn = rd.dot(nl)
            cos2t = 1 - nnt * nnt * (1 - ddn * ddn)
            if cos2t < 0:
                emissions[sid, depth] = e
                brdfs[sid, depth] = f
                rp, rd = reflRay
                continue
            tdir = rd * nnt - n * (1 if into else -1) * (ddn * nnt + ti.sqrt(cos2t))
            tdir = tdir.normalized()
            a = nt - nc
            b = nt + nc
            R0 = a * a / (b * b)
            c = 1 - (-ddn if into else ti.dot(tdir, n))
            Re = R0 + (1 - R0) * c ** 5
            Tr = 1 - Re
            P = .25 + .5 * Re
            RP = Re / P
            TP = Tr / (1 - P)

            if ti.random() < P:
                f *= RP
                rp, rd = reflRay
            else:
                f *= TP
                rp, rd = x, tdir

            emissions[sid, depth] = e
            brdfs[sid, depth] = f
            continue

    res = ti.Vector([0., 0., 0.])
    for i in range(DEPTH):
        ri = DEPTH - i - 1
        res *= brdfs[sid, ri]
        res += emissions[sid, ri]

    # test[2, 0] = res[0]
    # test[2, 1] = res[1]
    # test[2, 2] = res[2]
    # print(test)
    return res


@ti.kernel
def main():
    cam = ti.Vector([50, 52, 295.6]), ti.Vector([0, -0.042612, -1]).normalized()
    cx = ti.Vector([WIDTH * .5135 / HEIGHT, 0, 0])
    cy = cx.cross(cam[1]).normalized() * .5135

    ti.block_dim(64)

    for sid in range(WIDTH * HEIGHT * 4 * SAMPLES):
        tmp = sid
        s = tmp % SAMPLES
        tmp = tmp // SAMPLES
        sx = tmp % 2
        tmp = tmp // 2
        sy = tmp % 2
        tmp = tmp // 2
        x = tmp % WIDTH
        tmp = tmp // WIDTH
        y = tmp % HEIGHT

        r1 = 2 * ti.random()
        dx = ti.sqrt(r1) - 1 if r1 < 1 else 1 - ti.sqrt(2 - r1)
        r2 = 2 * ti.random()
        dy = ti.sqrt(r2) - 1 if r2 < 1 else 1 - ti.sqrt(2 - r2)
        d = cx * (((sx + .5 + dx) / 2 + x) / WIDTH - .5) + \
            cy * (((sy + .5 + dy) / 2 + y) / HEIGHT - .5) + cam[1]
        r = radiance(cam[0] + d * 140, d.normalized(), sid) * (1. / SAMPLES)

        i = (HEIGHT - y - 1) * WIDTH + x
        colors[i, sx, sy, s] = ti.Vector([clamp(r[0]), clamp(r[1]), clamp(r[2])]) * .25

    # for y in ti.static(range(HEIGHT)):          # Loop over image rows
    #     # logging.error(f"\rRendering ({SAMPLES*4} spp) {100.*y/(HEIGHT-1)}%")
    #     for x in ti.static(range(WIDTH)):
    #         for sy in ti.static(range(2)):     # 2x2 subpixel rows
    #             i = (HEIGHT - y - 1) * WIDTH + x
    #             for sx in ti.static(range(2)):
    #                 r = ti.Vector([0, 0, 0])
    #                 for s in ti.static(range(SAMPLES)):
    #                     r1 = 2 * ti.random()
    #                     dx = ti.sqrt(r1) - 1 if r1 < 1 else 1 - ti.sqrt(2-r1)
    #                     r2 = 2 * ti.random()
    #                     dy = ti.sqrt(r2) - 1 if r2 < 1 else 1 - ti.sqrt(2 - r2)
    #                     d = cx * (((sx+.5 + dx)/2 + x)/WIDTH - .5) + \
    #                         cy * (((sy+.5 + dy)/2 + y)/HEIGHT - .5) + cam[1]
    #
    #                     sid = ((((0 * HEIGHT + y) * WIDTH + x) * 2 + sy) * 2 + sx) * SAMPLES + s
    #                     r = r + radiance(cam[0] + d * 140, d.normalized(), sid) * (1./SAMPLES)
    #                 colors[i] = colors[i] + ti.Vector([clamp(r[0]), clamp(r[1]), clamp(r[2])]) * .25


if __name__ == '__main__':
    init()
    main()
    c = colors.to_numpy()
    c = np.sum(c, (1, 2, 3))

    def to_int(x):
        return int(pow(x, 1 / 2.2) * 255 + .5)

    for v in c:
        v[0] = to_int(v[0])
        v[1] = to_int(v[1])
        v[2] = to_int(v[2])
    cv2.imwrite('ti_001.jpg', c.reshape([HEIGHT, WIDTH, 3]))
