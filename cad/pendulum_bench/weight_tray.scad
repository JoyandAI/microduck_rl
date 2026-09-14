// 独立摆锤台架 —— Part 3b: 端部砝码托 weight_tray.scad
// M6 中心螺柱穿砝码片(标准 20mm 孔 50/100/200g 铸铁片 或 自制的钢片),
// 托盘底 + 顶盖夹紧; 砝码总质量 = 实际称重(计入 m_tip)。

tray_d  = 48;      // 托盘直径
tray_t  = 4;       // 托盘厚
m6_d    = 6.6;     // M6 通孔
disk_h  = 10;      // 砝码片高度典型值(只画示意)
cap_d   = 30;      // 顶盖直径
cap_t   = 4;

module tray() {
    difference() {
        cylinder(d = tray_d, h = tray_t);
        cylinder(d = m6_d, h = tray_t + 1, center = true);
    }
}
module cap() {
    difference() {
        cylinder(d = cap_d, h = cap_t);
        cylinder(d = m6_d, h = cap_t + 1, center = true);
    }
}
tray();                              // 放到杆末端
translate([0, 0, disk_h - tray_t]) cap();
