#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : ti_pt2.py
@Author: Chen Yanzhen
@Date  : 2021/3/2 17:57
@Desc  : 
"""

import taichi as ti

ti.init(print_ir=False)

DIFF = 0
SPEC = 1
REFR = 2
PI = 3.1415927
INF = 1e10
D_TYPE = ti.float64


@ti.data_oriented
class Scene(object):

    @ti.func
    def intersect(self, rp, rd):
        """

        :return: t, albedo, normal, emission, reflection
        """
        return 0.2, ti.Vector([0.3, 0.4, 0.5]), ti.Vector([0, 1., 0]), ti.Vector([0.8, 0.7, 0.6]), DIFF


@ti.func
def Vec(x=0., y=0., z=0.):
    return ti.Vector([x, y, z])


@ti.data_oriented
class SphereScene(Scene):

    def __init__(self, num: int):
        self.num = num
        self.radius = ti.field(D_TYPE, shape=self.num)
        self.positions = ti.Vector.field(3, D_TYPE, shape=self.num)
        self.emissions = ti.Vector.field(3, D_TYPE, shape=self.num)
        self.colors = ti.Vector.field(3, D_TYPE, shape=self.num)
        self.refls = ti.field(ti.int8, shape=self.num)
        self.cur = ti.field(ti.int8, shape=1)

    @ti.func
    def edit_sphere(self, i, radius, pos, ems, color, refl):
        self.radius[i] = radius
        self.positions[i] = pos
        self.emissions[i] = ems
        self.colors[i] = color
        self.refls[i] = refl

    @ti.kernel
    def init(self):
        self.edit_sphere(0, 1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF),
        self.edit_sphere(1, 1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), SPEC),
        self.edit_sphere(2, 1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF),
        self.edit_sphere(3, 1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF),
        self.edit_sphere(4, 1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),
        self.edit_sphere(5, 1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF),
        self.edit_sphere(6, 16.5, Vec(27, 16.5, 47), Vec(), Vec(1., 1, 1) * .999, SPEC),
        self.edit_sphere(7, 16.5, Vec(73, 16.5, 78), Vec(), Vec(1., 1, 1) * .999, REFR),
        self.edit_sphere(8, 600, Vec(50, 681.6 - .27, 81.6), Vec(12., 12, 12), Vec(), DIFF)
        # self.edit_sphere(8, 8.5, Vec(50, - 12.5 + 81.6 - .27, 81.6), Vec(12., 12, 12), Vec(), DIFF)

    @ti.func
    def intersect_sphere(self, rp, rd, i):
        res = -1.
        eps = 1e-4
        op = self.positions[i] - rp
        b = op.dot(rd)
        det = b * b - op.dot(op) + self.radius[i] ** 2
        # if det < 0:
        #     res = -1.
        # else:
        #     det = ti.sqrt(det)
        #     eps = 1e-4
        #     t = b - det
        #     if t > eps:
        #         res = t
        #     else:
        #         t = b + det
        #         if t > eps:
        #             res = t

        if det > 0:
            det = ti.sqrt(det)
            res = b - det
            if res < eps:
                res = b + det
            if res < eps:
                res = -1.
        return res

    @ti.func
    def intersect(self, rp, rd):
        min_t = -1.
        hid = 4
        for i in range(self.num):
            t = self.intersect_sphere(rp, rd, i)
            if min_t < 0 or 0 < t < min_t:
                min_t = t
                hid = i
        x = rp + rd * min_t
        normal = (x - self.positions[hid]).normalized()
        return min_t, self.colors[hid], normal, self.emissions[hid], self.refls[hid]


@ti.func
def clamp(x):
    return 0 if x < 0 else 1 if x > 1 else x


@ti.data_oriented
class Tracer(object):

    def __init__(self, w: int, h: int, samples: int, depth: int, scene: Scene):
        self.w = w
        self.h = h
        self.samples = samples
        self.depth = depth
        self.colors = ti.Vector.field(3, D_TYPE, shape=(w * h))
        self.scene = scene

        self.black_mask = ti.Vector.field(depth, D_TYPE, shape=depth)
        self.white_mask = ti.Vector.field(depth, D_TYPE, shape=depth)

    @ti.func
    def color_stack(self):
        return ti.Matrix([0.] * self.depth), ti.Matrix([0.] * self.depth), ti.Matrix([0.] * self.depth)

    @ti.func
    def add(self, c):
        return c + 1

    @ti.func
    def refl_diff(self, nl):
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

        return d

    @ti.func
    def refl_refr(self, n, nl, rd):
        new_rd = rd - n * 2 * n.dot(rd)
        coef = 1.
        into = n.dot(nl) > 0
        nc = 1
        nt = 1.5
        nnt = nc / nt if into else nt / nc
        ddn = rd.dot(nl)
        cos2t = 1 - nnt * nnt * (1 - ddn * ddn)
        if cos2t >= 0:
            tdir = rd * nnt - n * (1 if into else -1) * (ddn * nnt + ti.sqrt(cos2t))
            tdir = tdir.normalized()
            a = nt - nc
            b = nt + nc
            R0 = a * a / (b * b)
            c = 1 - (-ddn if into else tdir.dot(n))
            Re = R0 + (1 - R0) * c ** 5
            Tr = 1 - Re
            P = .25 + .5 * Re
            RP = Re / P
            TP = Tr / (1 - P)

            if ti.random() < P:
                coef = RP
                new_rd = rd - n * 2 * n.dot(rd)
            else:
                coef = TP
                new_rd = tdir
        return coef, new_rd

    @ti.func
    def radiance(self, rp, rd):

        es = self.color_stack()
        fs = self.color_stack()

        c = 0
        for depth in range(self.depth):
            t, f, n, e, refl = self.scene.intersect(rp, rd)
            if 0 < t < INF:
                c = self.add(c)
                x = rp + rd * t
                nl = n if n.dot(rd) < 0 else -n

                if depth == self.depth - 1:     # max iter
                    for ch in ti.static(range(3)):
                        es[ch] = es[ch] * self.white_mask[depth] + e[ch] * self.black_mask[depth]
                        fs[ch] = fs[ch] * self.white_mask[depth]
                else:
                    # if depth > 5:               # random stop
                    #     p = f.max()
                    #     if ti.random() < p:
                    #         f = f / p
                    #     else:
                    #         self.stack_ef(depth, es, fs, e, ti.Vector([0, 0, 0.]))
                    rp = x

                    if refl == DIFF:
                        rd = self.refl_diff(nl)
                    elif refl == SPEC:
                        rd = rd - n * 2 * n.dot(rd)
                    elif refl == REFR:
                        coef, rd = self.refl_refr(n, nl, rd)
                        f *= coef
                    for ch in ti.static(range(3)):
                        es[ch] = es[ch] * self.white_mask[depth] + e[ch] * self.black_mask[depth]
                        fs[ch] = fs[ch] * self.white_mask[depth] + f[ch] * self.black_mask[depth]

        res = ti.Vector([0., 0., 0.])
        # print(c)
        for i in ti.static(range(self.depth)):
            for ch in ti.static(range(3)):
                res[ch] *= fs[ch][self.depth - i - 1]
                res[ch] += es[ch][self.depth - i - 1]
        return res

    @ti.func
    def init_mask(self):
        for i in self.black_mask:
            self.black_mask[i] = ti.Vector([0.] * self.depth)
        for i in self.white_mask:
            self.white_mask[i] = ti.Vector([1.] * self.depth)
        for i in ti.static(range(self.depth)):
            self.black_mask[i][i] = 1.
            self.white_mask[i][i] = 0.

    @ti.kernel
    def trace(self, cam_x: float, cam_y: float, cam_z: float, cam_dir_x: float, cam_dir_y: float, cam_dir_z: float):
        self.init_mask()

        cam_pos = ti.Vector([cam_x, cam_y, cam_z])
        cam_dir = ti.Vector([cam_dir_x, cam_dir_y, cam_dir_z]).normalized()
        cx = ti.Vector([self.w * .5135 / self.h, 0, 0])
        cy = cx.cross(cam_dir).normalized() * .5135

        ti.block_dim(64)
        for pid in range(self.w * self.h):
            x = pid % self.w
            y = pid // self.w
            color = ti.Vector([0., 0., 0.])

            for sid in range(4 * self.samples):
                tmp = sid
                s = tmp % self.samples
                tmp = tmp // self.samples
                sx = tmp % 2
                tmp = tmp // 2
                sy = tmp % 2

                r1 = 2 * ti.random()
                dx = ti.sqrt(r1) - 1 if r1 < 1 else 1 - ti.sqrt(2 - r1)
                r2 = 2 * ti.random()
                dy = ti.sqrt(r2) - 1 if r2 < 1 else 1 - ti.sqrt(2 - r2)
                d = cx * (((sx + .5 + dx) / 2 + x) / self.w - .5) + \
                    cy * (((sy + .5 + dy) / 2 + y) / self.h - .5) + cam_dir
                r = self.radiance(cam_pos + d * 140, d.normalized()) * (1. / self.samples)

                color += ti.Vector([clamp(r[0]), clamp(r[1]), clamp(r[2])]) * .25
            i = (self.h - y - 1) * self.w + x
            self.colors[i] = color

    def image(self):
        c = self.colors.to_numpy()

        def to_int(x):
            return int(pow(x, 1 / 2.2) * 255 + .5)

        for v in c:
            v[0] = to_int(v[0])
            v[1] = to_int(v[1])
            v[2] = to_int(v[2])

        return c.reshape([self.h, self.w, 3])


if __name__ == '__main__':
    import cv2
    import time
    st = time.time()
    scene = SphereScene(9)
    tracer = Tracer(1600, 1200, 100, 10, scene)
    scene.init()
    tracer.trace(50, 52, 295.6, 0, -0.042612, -1)
    cv2.imwrite('ti2_005.jpg', tracer.image())
    ed = time.time()
    print("[TIME]", ed - st)

