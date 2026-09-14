// 独立摆锤台架 —— Part 2: U 型舵机夹持座 servo_clamp.scad
// 夹住 HL-2909 机身(34 宽 × 20 高 × 23 深, 输出轴沿 34mm 方向两端伸出)。
// 左侧(花键输出端)开 Ø8 让位槽; 右侧(对侧圆轴)开 Ø7 让位槽;
// 侧面 M3 顶丝压紧机身; 底部 M4×2 锁底座。
// 材质: 铝块(推荐) / PETG 打印(3 层壁 + 40% 填充, 顶丝加 M3 铜螺母)。

servo_w = 34;     // 机身宽(X 向, 轴方向)
servo_h = 20;     // 机身高
servo_d = 23;     // 机身深(Y)

wall    = 4;      // 壁厚
bot_t   = 4;      // 底部板厚
ridge   = 6;      // 上唇压板高
clr     = 1;      // 装配间隙

block_w = servo_w + 2 * wall;         // 44
block_d = servo_d + 2 * wall;         // 31
block_h = bot_t + servo_h + clr + ridge;  // 31

screw_d = 4.4;    // M4 底座孔
clamp_d = 3.4;    // M3 顶丝孔

difference() {
    cube([block_w, block_d, block_h]);
    // 内腔
    translate([wall, wall, bot_t])
        cube([servo_w + clr, servo_d + clr, servo_h + clr + 1]);
    // 左: 花键输出让位槽 Ø8(贯穿上唇前段)
    translate([-1, block_d / 2, bot_t + (servo_h + clr) / 2])
        rotate([0, 90, 0]) cylinder(h = wall + 2, d = 8, center = true);
    // 右: 对侧圆轴让位槽 Ø7
    translate([block_w - wall - 1, block_d / 2, bot_t + (servo_h + clr) / 2])
        rotate([0, 90, 0]) cylinder(h = wall + 2, d = 7, center = true);
    // 顶丝孔(贯穿右壁 → 顶住机身)
    translate([block_w - wall / 2, block_d / 2, bot_t + (servo_h + clr) / 2])
        rotate([0, 90, 0]) cylinder(h = wall + 2, d = clamp_d, center = true);
    // 底座孔 ×2(M4 沉头, 与 base_plate 孔距一致)
    for (sx = [-1, 1])
        translate([sx * 30, 0, -1]) cylinder(h = bot_t + 2, d = screw_d);
}
echo(str("外廓 = ", block_w, " x ", block_d, " x ", block_h, " mm"));
