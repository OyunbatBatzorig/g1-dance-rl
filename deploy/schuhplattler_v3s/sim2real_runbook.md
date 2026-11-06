# Sim2real: Schuhplattler v3s policy-г бодит G1 дээр ажиллуулах

Policy: `sch_v3s_floor / model_34998.pt` (full action, DR v2 + DR+ + floor mix).
Deploy багц: `~/projects/g1_pilot/deploy/schuhplattler_v3s/` (Unitree `config/policy/mimic/dance_102/`-той ижил бүтэцтэй).

## Төлөв

| Алхам | Робот хэрэгтэй юу | Төлөв |
|---|---|---|
| 1. ONNX export, deploy багц, joint дараалал | үгүй | ✅ дууссан |
| 2. Deploy логикийг MuJoCo дээр ONNX-ээр ажиллуулах | үгүй | ✅ Python хуулбараар хийсэн |
| 2б. Жинхэнэ C++ `g1_ctrl` + `unitree_mujoco` | үгүй (keyboard gamepad) | ✅ дууссан |
| 3. Бодит G1: gantry → бүтэн хурд | тийм | ⏳ төлөвлөгөө доор |

## 1. Export ба шалгалт (хийсэн)

```bash
cd ~/projects/g1_pilot
CK=~/projects/unitree_rl_lab/logs/rsl_rl/g1_schuhplattler_residual/2026-10-31_22-40-36_sch_v3s_floor/model_34998.pt
conda run -n env_isaaclab python deploy/export_onnx.py export $CK deploy/schuhplattler_v3s/exported/policy.onnx
conda run -n s2s python deploy/export_onnx.py check $CK deploy/schuhplattler_v3s/exported/policy.onnx
conda run -n unitree_sim_env python deploy/build_deploy_pkg.py
```

- ONNX: оролт `obs [1,154]`, гаралт `actions [1,29]`. Нэр, хэмжээ нь Unitree dance_102-той ижил. ONNX ба PyTorch-ийн хамгийн их зөрөө 7.6e-6.
- Joint дараалал гурван аргаар таарсан: `joint_ids_map`, MJCF (= SDK motor) нэрс, `day1/joint_map.csv`.
  - `stiffness`/`damping` нь **motor (SDK) дарааллаар** хадгалагддаг. C++ нь `kp[i] = stiffness[i]` гэж шууд онооно.
  - `scale`/`offset`/`default_joint_pos` нь **policy (Isaac) дарааллаар** хадгалагддаг.
- Сургалтын үед гарсан `deploy.yaml`-ийг дараах байдлаар цэвэрлэсэн:
  - `default_joint_pos` ба `offset`: 0-р роботын санамсаргүйгээр өөрчлөгдсөн (±0.03 rad) утгыг нэрлэсэн утгаар сольсон. Unitree-ийн өөрийн dance_102 файлд ч ийм алдаа байна.
  - C++ уншдаггүй residual талбарууд болон `!!python/tuple` tag-ийг хассан.
- Хөдөлгөөний CSV 50 Hz-ээр гарсан. Ингэснээр policy-ийн алхам бүр frame дээр яг таардаг. Үүнээс гадна эхэндээ 2 сек-ийн зөөлөн шилжилттэй хувилбар бий: `_intro2.csv`. Энэ хэсэгт робот зогсож байгаа байрлалаасаа (гар доош) бүжгийн эхний байрлал руу (гар хажуу тийш) шилжинэ.

## 2. Deploy логик MuJoCo дээр (хийсэн)

`deploy/deploy_sim2sim.py` нь C++ `State_Mimic.cpp`-ийн логикийг Python-оор давтдаг:
- CSV уншигч: `round` + blend, хурдыг forward difference-ээр тооцно.
- Торсыг pelvis IMU × waist joint-уудаас тооцно.
- `init_quat`-аар yaw-г тохируулна.
- PD-г motor бүрийн kp/kd-ээр, ONNX-ийг onnxruntime 1.22-оор (deploy-тэй ижил) ажиллуулна.
- DR болон MuJoCo загвар нь `sim2sim_dance.py`-тэй ижил, ижил seed нь ижил DR өгнө.

| Тест (512 run, сургалтын DR) | Дуусгасан | Бодит уналт |
|---|---|---|
| PyTorch sim2sim (`sim2sim_dance.py`), лавлагаан дээрээс эхэлсэн | 97.9% | 0.0% |
| **Deploy + ONNX**, лавлагаан дээрээс эхэлсэн | **99.2%** | **0.0%** |
| Deploy + ONNX, DR ×1.5 | 96.3% | 1.0% |
| Deploy + ONNX, зогсож байгаа байрлалаас, intro-гүй | 98.8% | 0.4% |
| **Deploy + ONNX, зогсож байгаа байрлалаас, 2 сек intro** | **99.4%** | **0.0%** |

Nominal body err: PyTorch 4.4 cm, deploy 4.4 cm. Тиймээс deploy-ийн obs (forward-difference хурд, waist-аас тооцсон торс) policy-д нөлөөлөхгүй байна.

```bash
cd ~/projects/g1_pilot
bash deploy/run_checks.sh ref      # лавлагаан дээрээс эхэлсэн, 1× ба DR ×1.5
bash deploy/run_checks.sh stand    # зогсож байгаа байрлалаас, intro, хурд 1.0 / 0.9 / 0.75 / 0.5
```

### 2б. Жинхэнэ C++ кодоор, keyboard gamepad-аар (хийсэн)

Xbox gamepad байхгүй тул `unitree_mujoco`-гийн хуулбарт keyboard gamepad нэмсэн: тоон товч бүр gamepad-ийн нэг комбо болно.
- Хоёр програм хоёулаа `~/projects/g1_pilot/sim_cpp/` дотор хуулбараар байгаа, анхны repo-нууд өөрчлөгдөөгүй:
  - `simulate/`: keyboard gamepad-тай `unitree_mujoco`
  - `deploy/robots/g1_29dof/`: Schuhplattler болон ArmTest төлөвтэй `g1_ctrl`
- Багц C++ дээр алдаагүй ачаалагдсан:
  - `Loaded motion file 'schuhplattler_gmr_v3s_50hz_intro2' with duration 16.98s`
  - `… 'schuhplattler_v3s_arm_test_50hz' with duration 7.00s`

```bash
bash ~/projects/g1_pilot/sim_cpp/run_sim.sh          # гараар: MuJoCo цонхон дээр дараад товчоор удирдана
bash ~/projects/g1_pilot/sim_cpp/run_sim.sh auto     # автомат: босгох → arm test → бүтэн бүжиг
```

| Товч | Gamepad комбо | Шилжилт |
|---|---|---|
| `1` | LT + up | Passive → FixStand |
| `8` ×5 | | elastic band-ийг 0.5 m буулгах (хөл шалан дээр) |
| `2` | RB + X | FixStand → Velocity |
| `9` | | elastic band унтраах |
| `3` | LT(2s) + up | Velocity → Mimic_Schuhplattler_ArmTest (7 сек → Velocity) |
| `4` | LT(2s) + right | Velocity → Mimic_Schuhplattler (16.96 сек → Velocity) |
| `0` | LT + B | → Passive (яаралтай зогсоолт) |

Автомат тестийн үр дүн (2026-11-01):
- FSM-ийн шилжилт бүхэлдээ ажилласан: Passive → FixStand → Velocity → ArmTest (7.0 сек) → Velocity → Schuhplattler (16.96 сек) → Velocity.
- Робот бүх хугацаанд босоо байсан: аарцгийн өндөр ≥ 0.756 m, их биеийн хамгийн их хазайлт 19°.

## Удаан хурдаар туршвал нэг хөл дээрх тэнцвэр асуудалгүй юу?

**Асуудалтай. Удаашруулах нь аюулгүй болгохгүй, харин илүү эрсдэлтэй болгоно.** (`config.yaml`-ийн `fps`-ийг багасгаж удаашруулсан. Бусад нь ижил, 512 run.)

| Тоглуулах хурд (`fps`) | Дуусгасан | Бодит уналт |
|---|---|---|
| 1.0× (50) | 99.4% | 0.0% |
| 0.9× (45) | 96.9% | 1.8% |
| 0.75× (37.5) | 57.8% | 23.2% |
| 0.5× (25) | 0.0% | 99.8% |

Шалтгаан (`deploy/balance_check.py`):
- Бүжгийн 67% нь нэг хөл дээр явагддаг. Хамгийн урт хэсгүүд нь 1.6–1.9 сек: 3.5–5.3, 5.4–7.2, 7.3–9.2, 9.2–10.8 сек.
- Нэг хөл дээр байх frame-уудын зөвхөн **63%**-д л массын төв тулгуур хөлийн дээр байна. Хамгийн их зөрүү 7.7 cm.
- Өөрөөр хэлбэл бүжиг статик тэнцвэртэй биш, хөдөлгөөний хурд (динамик) дээр тулгуурладаг. Удаан тоглуулахад тэр нөлөө алга болж, робот хажуу тийш унадаг. 0.5×-д уналт бүгд нэг хөл дээр зогсох эхний урт хэсэгт гардаг.
- Policy зөвхөн 1× хурдаар сурсан. Удаан хурдтай хөдөлгөөн түүний хувьд шинэ нөхцөл (out-of-distribution) юм.

**Санал:**
- Бодит туршилтыг **бүтэн (1×) хурдаар** хийх. Аюулгүй байдлыг удаан хурдаар биш, gantry болон богино хэсэг (`time_end`) ашиглаж хангах.
- Хэрэв заавал удаан хувилбар хэрэгтэй бол хурдыг 0.5–1.0 хооронд санамсаргүй өөрчилж (time-scale augmentation) шинээр fine-tune хийх шаардлагатай.

## 3. Бодит G1 дээрх төлөвлөгөө

### Бэлтгэл (робот ирэхээс өмнө)
- [x] 2б: C++ `g1_ctrl` + `unitree_mujoco` дээр бүх шилжилтийг шалгасан (keyboard gamepad, `sim_cpp/run_sim.sh`).
- [ ] Бүжиглэх шалны үрэлтийг мэдэх. μ ≥ 0.4 байвал MuJoCo-д 100% амжилттай. μ ≤ 0.3 буюу өнгөлсөн, нойтон шалан дээр бүү турш.
- [ ] Gantry, дэвсгэр, хамгаалах бүс. Яаралтай зогсоох хүн нь LT + B (Passive)-г мэдэж байх.

### Үе шат A: gantry дээр, хөл газарт хүрэхгүй
1. `g1_ctrl --network eth0` (робот дээрх анхны control програмыг унтраасан байх). Passive → FixStand.
2. Хөдөлгөөний чиглэлийг шалгах: joint бүр зөв тийшээ хөдөлж байгаа эсэх. Энэ нь joint дараалал болон тэмдгийн (sign) хамгийн сүүлийн шалгалт.

### Үе шат B: gantry-тай, хөл шалан дээр, олс сул
1. FixStand → Velocity. Робот өөрөө зогсож байх ёстой.
2. **Гарын smoke test** (`Mimic_Schuhplattler_ArmTest`, LT(2s) + up, 7 сек): хөл, бэлхүүс зогсох байрлалдаа үлдэж, зөвхөн гар хөдөлнө.
   - 1 сек хөдөлгөөнгүй → 2 сек бүжгийн эхний байрлал руу өргөнө → 1 сек барина → 2 сек буулгана → 1 сек хөдөлгөөнгүй.
   - Ингэснээр joint холболт, чиглэл, командын хурдыг бүжиглэхээс өмнө шалгана.
   - MuJoCo-д (512 run): 100% амжилттай, бодит уналт 0.
     - Nominal нөхцөлд хамгийн хурдан joint ~1 rad/s, мотор хүчний хязгаарынхаа ≤28%-ийг ашигласан.
     - Бүтэн бүжигт энэ нь 11 rad/s, 87% хүрдэг.
     - DR-тэй run-уудын ~9%-д тэнцвэрээ засахын тулд хөл хурдан хөдөлдөг (≥3 rad/s), гэхдээ унаагүй.
3. **Хэсэгчилсэн бүжиг** (`bash sim_cpp/set_dance.sh part`, 6.66 сек): 2 сек intro → бүжгийн эхний 1.66 сек (хоёр хөл шалан дээр) → 2 сек-д зогсох байрлал руу зөөлөн буцна → 1 сек хөдөлгөөнгүй → Velocity.
   - Бүтэн CSV-ийг `time_end`-ээр эрт зогсоох нь аюултай. Velocity policy гарыг огцом буулгадаг: 3.66 сек-д гар 2.7 rad (~157°) толгойн дээгүүр өргөгдсөн байдаг. Үүнийг C++ sim дээр 2026-11-02-нд харсан.
   - Хэсэгчилсэн хувилбар нь өөрөө зогсох байрлалаар төгсдөг:
     - deploy_sim2sim (512 run): 99.8% амжилттай, 1 удаа бодитоор унасан.
     - C++ sim: Velocity руу шилжсэний дараах хазайлт 0.8°.
   - 11.04 сек-ээр ижил хувилбар хийж туршихад робот тэр мөчид хурдан хөдөлж байдаг тул 29% нь бодитоор унасан. Тиймээс тэр хувилбарыг хассан.
4. **Бүтэн бүжиг** (`bash sim_cpp/set_dance.sh full`, 16.96 сек). Зогсох байрлалаар төгсдөг.
   - `run_real.sh` зөвхөн эдгээр хоёр тохиргоог (part / full) хүлээж авна.
6. Туршилт бүрийг видеогоор бичиж, `lowstate` лог хадгалах (`State_Mimic`-д лог бичигч нэмж болно).

### Үе шат C: gantry-гүй
- Үе шат B бүтэн бүжгээр 3–5 удаа дараалан амжилттай болсны дараа л gantry-г авна. Ойролцоо хамгаалах хүн байлгана.

### Зогсоох нөхцөлүүд
- Их бие 1 rad-аас илүү хазайвал (`bad_orientation`) C++ өөрөө Passive руу шилжинэ.
- Мотор халах, хэвийн бус чичиргээ, хөл гулсах тохиолдолд шууд LT + B дарна.
- Нэг хэсэгт хоёр удаа дараалан бүтэлгүйтвэл тэр хэсгийг MuJoCo болон Isaac-д дахин шинжилнэ.


## 4. Бодит робот дээр ажиллуулах командууд

`sim_cpp/deploy/robots/g1_29dof/build/g1_ctrl` нь sim дээр туршсан яг тэр binary, тохиргоо юм. Бодит робот руу зөвхөн `--network`-ийг солино.

1. **Сүлжээ:** роботыг энэ PC-тэй Ethernet кабелиар холбоно (`enp5s0` эсвэл `enp6s0`). Дараа нь роботын 192.168.123.x сүлжээнд статик IP өгнө (sudo, нэг удаа):
   ```bash
   sudo ip link set enp5s0 up
   sudo ip addr add 192.168.123.99/24 dev enp5s0
   ping -c 3 192.168.123.161
   ```
   `.161` бол Unitree-ийн удирдлагын компьютерын ердийн хаяг. Хариу ирэхгүй бол роботын гарын авлагаас хаягийг шалгана.
2. **Роботын өөрийн хөдөлгөөний програмыг унтраах (debug mode).** g1_ctrl моторуудад шууд команд илгээдэг. Роботын анхны locomotion програм зэрэг ажиллаж байвал хоёр удирдлага зөрчилдөнө. Unitree-ийн гарын авлагын дагуу damping / debug mode-д оруулна (G1-д ихэвчлэн remote-оор L2 + R2). Энэ товчлолыг заавал гарын авлагаас нягтална.
3. **Юуг тоглуулахыг сонгох** (g1_ctrl-ийг дахин эхлүүлэх үед хүчинтэй болно):
   ```bash
   bash ~/projects/g1_pilot/sim_cpp/set_dance.sh part     # эхний туршилт: 6.66 сек, зөөлөн буцалттай
   bash ~/projects/g1_pilot/sim_cpp/set_dance.sh full     # дараа нь: бүтэн бүжиг
   ```
   `set_time_end.sh` нь зөвхөн sim дээр туршихад зориулагдсан.
4. **Ажиллуулах.** Скрипт interface, sim ажиллаж байгаа эсэхийг шалгаж, checklist харуулж, "yes" гэж бичихийг хүлээнэ:
   ```bash
   bash ~/projects/g1_pilot/sim_cpp/run_real.sh enp5s0
   ```
5. **Remote** (Unitree-ийн remote; LT = L2, RB = R1):

| Remote | Шилжилт |
|---|---|
| L2 + up | Passive → FixStand (gantry дээр) |
| R1 + X | FixStand → Velocity |
| L2-г 2 сек барьж + up | Velocity → ArmTest (7 сек) |
| L2-г 2 сек барьж + right | Velocity → Schuhplattler (part эсвэл full, `set_dance.sh`-ээр сонгоно) |
| **L2 + B** | **аль ч горимоос → Passive (яаралтай зогсоолт)** |

Лог: `sim_cpp/logs/real_<огноо>.log`.

## Файлууд
| Файл | Үүрэг |
|---|---|
| `deploy/export_onnx.py` | checkpoint → ONNX, PyTorch-тэй харьцуулах |
| `deploy/build_deploy_pkg.py` | deploy багц (yaml, CSV, config snippet), joint дарааллын шалгалт |
| `deploy/deploy_sim2sim.py` | C++ State_Mimic-ийн Python хуулбар, MuJoCo, ONNX |
| `deploy/run_checks.sh` | 512 run-ийн тестүүд |
| `deploy/balance_check.py` | нэг хөл дээрх массын төв ба ZMP-ийн шинжилгээ |
| `deploy/results/*.json` | тестийн үр дүн |
| `deploy/schuhplattler_v3s/` | deploy багц (дотор нь гарын smoke test CSV) |
| `sim_cpp/run_sim.sh`, `run_real.sh`, `set_dance.sh`, `set_time_end.sh` | C++ sim (keyboard), бодит робот, `time_end` тохируулах |
