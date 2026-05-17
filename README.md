# 无人机路径规划实验项目

本项目包含两部分内容：

1. **二维路径规划实验**：用于比较不同初始化算法与优化方法；
2. **三维复杂障碍环境路径规划实验**：面向工业无人机，采用  
   **A\* 初始搜索 + B 样条初值平滑 + SQP 非线性优化** 的组合方法，在满足避障、速度、爬升率、高度和转弯角约束的前提下生成三维飞行路径。

三维部分的研究建模说明见：

```text
3_Dimensional_Q/3dimension.md
```

三维部分的实际代码见：

```text
3_Dimensional_Q/scripts/three_dimensional_path_planning.py
```

---

## 1. 环境准备

### 1.1 创建 Python 环境

```bash
conda create -n pathplan python=3.12
conda activate pathplan
```

### 1.2 安装 Python 依赖

```bash
pip install -r requirements.txt
```

当前核心依赖包括：

- `numpy`
- `scipy`
- `matplotlib`

### 1.3 可选：安装 ffmpeg

`ffmpeg` 只影响视频输出：

- 检测到 `ffmpeg` 时，可额外导出 `mp4`
- 未检测到时，脚本会自动退化为输出 `gif`

Ubuntu / Debian 可使用：

```bash
sudo apt update
sudo apt install ffmpeg
```

## 安装环境依赖
```bash
pip install -r requirements.txt
```

## 二维路径规划算法测试
---

## 2. 项目结构

```text
.
├── main.py
├── init_numerical_solution.py
├── meta_algorithm/
│   ├── a_star.py
│   ├── dijkstra.py
│   ├── rrt.py
│   ├── genetic.py
│   └── ant_colony.py
├── scripts/
│   └── path_planning.sh
└── 3_Dimensional_Q/
    ├── 3dimension.md
    ├── Drone_para.md
    ├── scripts/
    │   └── three_dimensional_path_planning.py
    └── results/
        └── three_dimensional/
            ├── three_dimensional_path_planning.png
            ├── three_dimensional_dashboard.png
            ├── three_dimensional_trajectory.gif
            ├── three_dimensional_metrics.json
            └── three_dimensional_waypoints.csv
```

---

## 3. 二维路径规划实验

二维部分沿用仓库原有实现，可通过统一脚本测试不同初始化算法与优化方法：

```bash
./scripts/path_planning.sh \
    --init_method "$INIT_METHOD" \
    --opt_method "$OPT_METHOD" \
    --waypoints "$WAYPOINTS"
# example: ./scripts/path_planning.sh --init_method dijkstra --opt_method penalty --waypoints 150
```

### 3.1 支持的初始化方法

```text
straight / dijkstra / a_star / rrt / genetic / aco
```

### 3.2 支持的优化方法

```text
penalty / sqp
```
| 参数 | 含义 | 可选 |
|---|---|---|
| `--init_method` | 初始路径规划算法 | `straight / dijkstra / a_star / rrt / genetic / aco` |
| `--opt_method` | 数值优化算法 | `penalty / sqp` |
| `--output-dir` | 输出目录 | `results/ & video/` |

## 三维路径规划算法测试
### 3.3 示例

```bash
python 3_Dimensional_Q/scripts/three_dimensional_path_planning.py \
    --segments 20 \
    --safety-margin 5
```
| 参数 | 含义 | 默认值 |
|---|---|---|
| `--segments` | 路径分段数 | `20` |
| `--safety-margin` | 障碍物外额外安全距离 | `5.0 m` |
| `--output-dir` | 输出目录 | `3_Dimensional_Q/results/three_dimensional` |



./scripts/path_planning.sh --init_method dijkstra --opt_method penalty --waypoints 150
```

---

## 4. 三维复杂环境路径规划

### 4.1 无人机选型

三维实验选择 **DJI Matrice 350 RTK** 作为建模对象。选择它的原因是：

- 属于工业级多旋翼无人机，适合城市巡检、山区巡检等复杂环境任务；
- 具备较明确的水平速度、垂直速度、续航等工程参数，便于转化为路径规划约束；
- 机体尺寸和轴距能够支持安全包络半径的近似建模；
- 其工作温度、防护等级和抗风能力更接近实际工程使用场景，而不只是理想化仿真对象。

#### 4.1.1 无人机参数

| 参数类别 | 参数 | 数值 / 说明 |
|---|---:|---|
| 机型 | DJI Matrice 350 RTK | 工业级多旋翼无人机 |
| 展开尺寸 | \(810 \times 670 \times 430\) mm | 不含桨叶 |
| 对角轴距 | 895 mm | 可用于近似安全包络半径 |
| 空机重量 | 约 3.77 kg | 不含电池 |
| 含双 TB65 电池重量 | 约 6.47 kg | 标准飞行状态 |
| 最大起飞重量 | 9.2 kg | 影响载荷与能耗 |
| 单云台最大载荷 | 960 g | 可用于载荷建模 |
| 最大飞行时间 | 55 min | 无载荷、无风、约 8 m/s 条件 |
| 最大水平速度 | 23 m/s | 在代码中作为速度约束 |
| 最大上升速度 | 6 m/s | 在代码中作为爬升约束 |
| 最大垂直下降速度 | 5 m/s | 在代码中作为下降约束 |
| 最大倾斜下降速度 | 7 m/s | 当前版本未单独建模 |
| 最大抗风速度 | 12 m/s | 当前版本未显式建模风场 |
| 最大飞行海拔 | 5000 m / 7000 m | 与桨叶和载荷有关 |
| 工作温度 | \(-20^\circ C \sim 50^\circ C\) | 环境适应性指标 |
| 防护等级 | IP55 | 适合复杂户外环境 |

#### 4.1.2 参数如何映射到代码

三维脚本中的 `PlannerConfig` 将无人机参数转化为优化约束：

| 代码参数 | 默认值 | 含义 |
|---|---:|---|
| `drone_radius` | `1.0 m` | 无人机等效安全半径 |
| `safety_margin` | `5.0 m` | 障碍物外的额外安全裕度 |
| `cruise_speed` | `8.0 m/s` | 用于估计总飞行时间与每段时间 |
| `max_speed` | `23.0 m/s` | 最大速度约束 |
| `max_climb_speed` | `6.0 m/s` | 最大爬升速度约束 |
| `max_descent_speed` | `5.0 m/s` | 最大下降速度约束 |
| `max_turn_deg` | `45°` | 相邻航段最大转弯角约束 |
| `max_flight_time_s` | `55 min` | 当前保留在配置中，暂未作为独立优化约束启用 |

说明：

- `drone_radius=1.0 m` 不是机体物理半径，而是为了工程安全留出的等效包络半径；
- 当前飞行时间不是优化变量，而是先根据起终点直线距离和 `cruise_speed=8.0 m/s` 估算总时长，再均分到各航段：
  \[
  \Delta t = \frac{\|P_g-P_s\| / v_{\text{cruise}}}{n_{\text{segments}}}
  \]
- 实际障碍物半径在规划时会膨胀为  
  \[
  R_{\text{inflated}} = R_{\text{obstacle}} + r_{\text{drone}} + d_{\text{safety}}
  \]
  以便把无人机尺寸和安全距离统一纳入避障约束。

---

### 4.2 三维实验场景

#### 4.2.1 空间范围

| 项目 | 设置 |
|---|---:|
| 三维空间范围 | \(500m \times 500m \times 100m\) |
| X 范围 | \(0 \sim 500m\) |
| Y 范围 | \(0 \sim 500m\) |
| Z 范围 | \(20 \sim 120m\) |
| 三维栅格分辨率 | \(10m \times 10m \times 5m\) |
| 路径分段数 | 20 |

#### 4.2.2 起点与终点

```text
起点 Ps = (0, 0, 30)
终点 Pg = (500, 420, 60)
```

#### 4.2.3 障碍物建模

当前版本使用 **球形障碍物**：

| 障碍物 | 球心坐标 \((x,y,z)\) | 原始半径 |
|---|---:|---:|
| O1 | \((120,100,40)\) | 45 m |
| O2 | \((240,180,70)\) | 60 m |
| O3 | \((330,300,50)\) | 55 m |
| O4 | \((410,360,80)\) | 50 m |

在求解时，障碍物会按无人机半径和安全距离进行膨胀处理。  
默认情况下，最终用于碰撞判断的半径为：

```text
原始障碍物半径 + 1 m 无人机等效半径 + 5 m 安全距离
```

---

### 4.3 三维算法总体流程

当前代码实现的是：

```text
A* 粗路径搜索
        ↓
按弧长降采样
        ↓
B 样条平滑初值
        ↓
SQP / SLSQP 非线性优化
        ↓
生成路径、指标、图像和动画
```

对应脚本：

```text
3_Dimensional_Q/scripts/three_dimensional_path_planning.py
```

路径采用离散航路点参数化。默认 `n_segments=20`，因此共有：

- 21 个航路点；
- 19 个内部待优化航路点；
- \(19 \times 3 = 57\) 个连续优化变量。

---

### 4.4 算法实现细节

#### 4.4.1 第一步：三维 A\* 搜索可行粗路径

函数：

```python
astar_3d(config)
```

实现逻辑：

1. 将三维空间离散为规则网格；
2. 对障碍物执行安全膨胀；
3. 每个网格节点最多向 26 个邻居扩展；
4. 使用欧氏距离作为启发函数：
   \[
   f(n)=g(n)+h(n)
   \]
5. 找到从起点到终点的无碰撞粗路径。

这里的 A\* 不追求最终最优，而是解决连续优化中非常关键的 **初值问题**：

- 避免直接用直线初始化时穿过障碍物；
- 降低 SQP 从不可行区域出发而失败的概率；
- 为后续平滑和优化提供一个结构合理的路径骨架。

---

#### 4.4.2 第二步：按弧长降采样

函数：

```python
resample_by_arclength(path, n_points)
```

A\* 生成的点数通常较多，不适合直接作为连续优化变量。  
代码会按照累计弧长均匀采样，将路径压缩为：

```text
n_segments + 1 = 21 个航路点
```

这样做有两个目的：

- 保留粗路径的整体绕障拓扑；
- 控制优化变量规模，使 SQP 更容易收敛。

---

#### 4.4.3 第三步：B 样条生成平滑初值

函数：

```python
bspline_resample(path, n_points)
```

代码会先对降采样路径进行三次 B 样条拟合，再重新采样得到更平滑的初始路径。

需要注意：

- **当前实现里，B 样条只用于生成优化初值；**
- **最终被 SQP 优化的变量仍然是离散航路点坐标，而不是 B 样条控制点。**

这是一种偏工程化的实现方式：先获得比 A\* 更平滑的起点，再用带约束的非线性优化做最终修正。

---

#### 4.4.4 第四步：构造优化目标函数

函数：

```python
objective(x, config, safety_margin)
```

综合目标函数由四部分组成：

\[
J = w_L J_L + w_S J_S + w_E J_E + w_O J_O
\]

##### 1. 路径长度项

\[
J_L = \sum_i \|P_{i+1}-P_i\|
\]

用于鼓励路径尽量短。

##### 2. 平滑性项

\[
J_S = \sum_i \|P_{i+1}-2P_i+P_{i-1}\|^2
\]

通过二阶差分惩罚突变，减少急转弯和路径抖动。

##### 3. 能耗近似项

代码中使用：

- 速度平方项；
- 正向爬升代价；
- 相邻速度变化代价。

其中爬升部分使用 `softplus` 代替不可导的 `max(0, vz)`，让目标函数保持平滑。

##### 4. 障碍物软惩罚项

对进入安全包络的点施加 `softplus` 惩罚，使优化器在接近障碍物时自动提高代价。

默认权重为：

| 项 | 权重 |
|---|---:|
| 路径长度 | `1.0` |
| 平滑性 | `0.02` |
| 能耗 | `0.01` |
| 障碍物软惩罚 | `20.0` |

---

#### 4.4.5 第五步：显式非线性约束

函数：

```python
nonlinear_constraints(x, config, safety_margin)
```

当前代码显式约束包括：

| 约束 | 代码含义 |
|---|---|
| 最大速度 | 每段距离不能超过 `max_speed × dt` |
| 最大爬升速度 | `dz <= max_climb_speed × dt` |
| 最大下降速度 | `-dz <= max_descent_speed × dt` |
| 高度范围 | 通过变量边界保证 `20m <= z <= 120m` |
| 转弯角 | 相邻航段夹角不得超过 `45°` |
| 避障 | 每条路径线段都必须在膨胀障碍物外 |

##### 线段级避障，而不是只检查航路点

这是当前实现中一个很重要的工程细节。

如果只检查航路点，可能出现：

```text
两个航路点都在障碍物外，但它们之间的连线穿过障碍物
```

因此代码使用：

```python
segment_min_dist_sq_to_center(...)
```

计算每条路径线段到每个球心的最小距离，并要求整段都满足安全半径约束。这比只对离散点做避障判断更可靠。

---

#### 4.4.6 第六步：同伦式 SQP 优化

函数：

```python
optimize_path(initial_path, config)
```

代码使用 `scipy.optimize.minimize(..., method="SLSQP")` 做 SQP 风格的非线性约束优化。

为了减小强非线性约束直接施加时的求解难度，代码采用逐步增大安全裕度的同伦策略：

```text
0.5 m  →  2.0 m  →  5.0 m
```

每一阶段都以上一阶段结果作为初值，再进入下一阶段。  
这样通常比“一开始就要求最终安全距离”更稳。

---

### 4.5 当前实现和研究方案的关系

`3dimension.md` 给出的是完整研究方案，当前脚本已经落地了其中的主干：

| 研究方案内容 | 当前代码状态 |
|---|---|
| 三维 A\* 初始路径 | 已实现 |
| 弧长降采样 | 已实现 |
| B 样条平滑 | 已实现，用于初值生成 |
| SQP 非线性优化 | 已实现，基于 `SLSQP` |
| 光滑避障惩罚 | 已实现，使用 `softplus` |
| 同伦式约束增强 | 已实现 |
| 速度 / 爬升 / 下降 / 转弯 / 高度约束 | 已实现 |
| 线段级连续避障检查 | 已实现，属于工程增强 |
| 稀疏雅可比矩阵 | 当前版本未显式实现 |
| B 样条控制点直接优化 | 当前版本未实现 |
| 最大飞行时间独立约束 | 当前版本未显式启用 |

这意味着当前实现已经足以完成三维路径规划实验和结果展示，但仍保留了后续继续研究的扩展空间。

---

### 4.6 运行三维实验

#### 4.6.1 默认运行

```bash
python 3_Dimensional_Q/scripts/three_dimensional_path_planning.py
```

如果使用仓库内虚拟环境：

```bash
./.venv/bin/python 3_Dimensional_Q/scripts/three_dimensional_path_planning.py
```

#### 4.6.2 自定义参数

```bash
python 3_Dimensional_Q/scripts/three_dimensional_path_planning.py \
    --segments 20 \
    --safety-margin 5
```

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--segments` | 路径分段数 | `20` |
| `--safety-margin` | 障碍物外额外安全距离 | `5.0 m` |
| `--output-dir` | 输出目录 | `3_Dimensional_Q/results/three_dimensional` |

---

### 4.7 三维实验输出

默认输出目录：

```text
3_Dimensional_Q/results/three_dimensional/
```

| 文件 | 说明 |
|---|---|
| `three_dimensional_path_planning.png` | 基础三维路径图 + XY 投影图 |
| `three_dimensional_dashboard.png` | 综合结果图：3D 轨迹、俯视图、高度、速度、净距 |
| `three_dimensional_trajectory.gif` | 无人机沿优化路径飞行动画 |
| `three_dimensional_trajectory.mp4` | 检测到 `ffmpeg` 时自动生成 |
| `three_dimensional_metrics.json` | 配置、指标、优化阶段、输出文件记录 |
| `three_dimensional_waypoints.csv` | 优化后的航路点坐标 |

---

### 4.8 一次默认运行的结果示例

基于当前默认参数，最近一次运行得到：

| 指标 | B 样条初始路径 | SQP 优化后路径 |
|---|---:|---:|
| 路径长度 | 697.700 m | 678.790 m |
| 平滑性指标 | 1136.143 | 80.245 |
| 最大速度 | 8.570 m/s | 9.034 m/s |
| 最大爬升速度 | 2.849 m/s | 2.037 m/s |
| 最大下降速度 | 2.880 m/s | 0.652 m/s |
| 最大转弯角 | 27.470° | 9.710° |

这组结果说明：

- 路径在满足安全约束的同时缩短了约 18.9 m；
- 平滑性显著改善，说明轨迹更适合实际飞行；
- 速度、爬升率、下降率和转弯角均保持在设定约束范围内；
- 最小安全净距接近 `0 m`，表示优化结果会主动贴近最终安全边界以换取更短路径。

---

## 5. 后续可扩展方向

如果继续扩展三维部分，比较自然的方向包括：

1. 将球形障碍物扩展为球体 + 圆柱体 + 建筑盒体的混合环境；
2. 把 B 样条控制点本身作为优化变量，而不只是用于初始化；
3. 引入风场、载荷、续航和能量模型；
4. 加入最大总飞行时间约束；
5. 使用稀疏雅可比或自定义梯度，加快大规模航路点优化；
6. 与 RRT\*、普通 SQP、仅 A\* 等方法进行系统对比实验。
