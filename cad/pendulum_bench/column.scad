// 独立摆锤台架 —— Part 2b: 立柱 column.scad
// 把 U 夹抬高到桌面上方, 使摆臂(≈L=200mm)能在桌沿外自由摆动:
// 轴心高 = 底座8 + 立柱230 + U夹底部4 + 半高10 = 252mm ≥ L+50。
// 顶部 4×M4 沉头(与 servo_clamp 底部孔距一致), 底部 2×M4 与底座相连。

col_h   = 230;
col_w   = 60;   // 深(Y), 与 U 夹近宽
col_t   = 24;   // 厚(X)

m4 = 4.6;
top_hole_sp = 25;   // 顶部两孔距(U 夹底部孔距)
bot_hole_sp = 30;   // 底部两孔距(装底座)

difference() {
    cube([col_t, col_w, col_h]);
    // 顶部 2×M4 通孔(贯穿 X, 接 U 夹)
    for (sy = [-1, 1])
        translate([col_t / 2, sy * top_hole_sp / 2 + 0, col_h - 3])
            rotate([0, 90, 0]) cylinder(d = m4, h = col_t + 2, center = true);
    // 底部 2×M4 沉头
    for (sy = [-1, 1])
        translate([col_t / 2, sy * bot_hole_sp / 2, 0])
            rotate([0, 90, 0]) cylinder(d = m4, h = col_t + 2, center = true);
}
echo(str("立柱 = ", col_t, " x ", col_w, " x ", col_h, " mm (轴心高≈252)"));
