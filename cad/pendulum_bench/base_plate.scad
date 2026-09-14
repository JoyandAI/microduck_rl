// 独立摆锤台架 —— Part 1: 底座板 base_plate.scad
// 材质建议: 铝板 8mm / 亚克力 8mm / PETG 打印(壁厚≥4层)
// 用法: 把参数改好后, openscad 打开 → Render → Export STL 打印;
//       或按尺寸表直接手绘铝板打孔(全部用标准钻头/沉头)。
// 单位: mm

base_len  = 120;   // 长(X)
base_wid  = 80;    // 宽(Y)
base_t    = 8;     // 厚
hole_d    = 4.4;   // M4 通孔(沉头 M4 用 8mm)
hole_sp_x = 88;    // 孔距 X
hole_sp_y = 52;    // 孔距 Y

difference() {
    cube([base_len, base_wid, base_t], center=true);
    for (sx = [-1, 1]) for (sy = [-1, 1])
        translate([sx * hole_sp_x / 2, sy * hole_sp_y / 2, base_t / 2])
            cylinder(h = base_t + 2, d = hole_d, center = true);
}
// 顶部贴合标记(可选, 便于对中, 不打印也行)
translate([0, 0, base_t/2]) cylinder(h = 0.6, d = 30);
