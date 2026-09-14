// 独立摆锤台架 —— Part 3a: 摆杆夹头 rod_clamp.scad
// 夹 Ø6 碳纤/铝杆; 通过 3×M3 沉头孔固定在"金属舵机臂"(随舵机附带的
// 25T 花键铜/铝臂, 选带 2 个 M3 孔、孔距 10-16mm 的型号)上。

rod_d  = 6;       // 杆直径
h_cl   = 8;       // 夹头高(Y)
w      = 18;      // 宽(X)
t      = 12;      // 厚(Z, 压杆方向)

m3_d   = 3.4;     // M3 孔
screw_sp = 12;    // 两 M3 孔距(按你的舵机臂实测改)

difference() {
    cube([w, h_cl, t]);
    // 杆孔(贯穿 Z)
    rotate([0, 90, 0]) cylinder(d = rod_d, h = w, center = true);
    // 锁紧缝: 从顶面切到杆孔(M3 压缝)
    translate([0, h_cl / 2, t * 0.62]) cube([w, 1.0, t * 0.4]);
    // 两 M3 螺纹孔(固定到舵机臂)
    for (sx = [-1, 1])
        translate([sx * screw_sp / 2, 0, -1]) cylinder(d = m3_d, h = 3);
    // 侧向 M3 锁杆孔
    translate([w / 2, h_cl / 2, 0]) rotate([0, 0, 0]) cylinder(d = m3_d, h = t + 2, center = true);
}
