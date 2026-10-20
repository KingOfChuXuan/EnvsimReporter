# EnvsimReporter 代码逐段讲解

> 对应文件：[EnvsimReporter.py](EnvsimReporter.py)（719 行，2026-10-20）  
> 建议阅读顺序：先看 §0 的数据流，再按函数顺序看。行号以当前文件为准。

---

## 0. 整体数据流

```text
OpenMM Simulation
      │  每 interval 步调用 report(simulation, state)
      ▼
state.getPositions()  →  只取水 O 的坐标 (N,3) [nm]
state.getPeriodicBoxVectors()  →  (3,3) [nm]
      │
      ▼
compute_environment_similarity()
      │
      ├─ 选邻居：i<j 的 pair 列表 (ci, cj, d)
      │    正交盒：cKDTree(boxsize)
      │    三斜盒：27 个镜像 + cKDTree
      │
      ├─ 对每个参考环境 m：
      │     d2_forward[p,l] = |d_p - r_l|²
      │     d2_backward[p,l] = |d_p + r_l|²
      │     kernel_forward[p] = Σ_l exp(-d2_forward/4σ²)/n
      │     kernel_backward[p] = Σ_l exp(-d2_backward/4σ²)/n
      │     values[m] = bincount(ci, kernel_forward)
      │                 + bincount(cj, kernel_backward)
      │
      ├─ 多参考 softmax：env = (1/λ) log Σ_m exp(λ values[m])
      ▼
env (N,)  →  env_mean / n_ice / 写文件
```

一句话：**reporter 把 PLUMED 每个 MD step 都在做的 CV 计算，搬到每个 report 帧只做一次；只算值，不进入力的链式法则。**

---

## 1. 文件头、import、常量（第 1–63 行）

```python
from openmm.unit import Unit, kelvin, nanometer, nanosecond
```

- `Unit`：用来检查 `time_unit` 必须是 `openmm.unit` 的单位对象（比如 `nanosecond`、`picosecond`）。
- `kelvin`：读温度时把 `getTemperature()` 的结果转成 K。
- `nanometer`：OpenMM 内部长度单位是 nm，坐标和盒子都转成 nm。
- `nanosecond`：输出时间的默认单位；不再在代码里用字符串 `'ns'/'ps'` 判断。

```python
_ANGSTROM_TO_NM = 0.1
```

参考 PDB 的坐标是 Å，PLUMED/OpenMM 内部用 nm，所以读入后乘 0.1。

```python
_DEFAULT_WATER_NAMES = ("HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE")
_DEFAULT_OXYGEN_NAMES = ("O", "OW", "OH2", "O1", "OT")
```

自动选水 O 时的候选 residue name 和原子名。优先用 topology 里的 element（O），element 缺失时退回原子名。

---

## 2. `read_reference_pdb()`（第 66–115 行）

作用：读一个 PLUMED 风格的参考环境 PDB，返回 `(n,3)` 的位移矢量（Å）。

```python
coords = []
in_model = False
model_seen = False
```

- `coords` 收集坐标；
- `in_model` 表示当前在第一个 MODEL 里；
- `model_seen` 表示已经见过 MODEL。

```python
record = line[:6].strip().upper()
```

PDB 的 record 名在前 6 列（`ATOM  `、`HETATM`、`MODEL `、`ENDMDL`）。`strip().upper()` 让判断不受空格/大小写影响。

```python
if record == "MODEL":
    if model_seen:
        break
    model_seen = True
    in_model = True
    continue
```

只读第一个 MODEL：第二次遇到 `MODEL` 就直接停止。

```python
if record == "ENDMDL":
    if in_model:
        break
    continue
```

第一个 MODEL 结束时停止。没有 MODEL 的文件（普通单帧 PDB）则读到 EOF。

```python
if record not in ("ATOM", "HETATM"):
    continue
```

忽略 `CRYST1`、`TER`、`CONECT` 等非坐标行。

```python
x = float(line[30:38])
y = float(line[38:46])
z = float(line[46:54])
```

PDB 标准坐标列：x=31–38，y=39–46，z=47–54（Python 0-based 切片是 30:38 等）。这里**按固定列解析**，因为参考 PDB 里 4 个原子可能共用同一个 serial；`openmm.app.PDBFile` 会按 serial 去重，把 4 原子参考读成 1 原子。

```python
coords.append((x, y, z))
...
arr = np.asarray(coords, dtype=np.float64)
```

最后转成 `(n,3)` 数组；空文件或非有限坐标会抛 `ValueError`。

---

## 3. 盒子工具（第 118–138 行）

### `_is_orthorhombic(box)`

```python
return bool(np.allclose(box, np.diag(np.diag(box)), atol=atol))
```

如果盒子矩阵的非对角元都接近 0，就是正交盒。正交盒可以走更快的 `cKDTree(boxsize=...)`。

### `_box_widths(box)`

```python
a, b, c = box
volume = abs(float(np.dot(a, np.cross(b, c))))
```

三个 box vector 是矩阵的三行；体积 = `a · (b × c)`。

```python
widths = [
    volume / |b × c|,
    volume / |c × a|,
    volume / |a × b|,
]
```

这是三组对面之间的**垂直宽度**。三斜盒的 27 镜像法要求 `r_cut` 小于最小垂直宽度，否则一个方向上的 ±1 镜像不够。

---

## 4. 邻居表：三种实现

三个函数都返回同样的三元组：

- `ci`：pair 的中心原子 index，shape `(P,)`
- `cj`：pair 的邻居原子 index，shape `(P,)`
- `d`：`r[cj] - r[ci]` 的最小镜像位移，shape `(P,3)`

并且只保留 `ci < cj`，每个无序 pair 出现一次。

### 4.1 正交盒 `_pairs_orthorhombic()`（第 141–170 行）

```python
n = len(positions)
lengths = np.diag(box)
```

正交盒只有三个长度。

```python
wrapped = positions % lengths
tree = cKDTree(wrapped, boxsize=lengths)
neighbours = tree.query_ball_point(wrapped, r_cut, workers=-1, return_sorted=False)
```

- `positions % lengths` 把坐标折回主盒 `[0,L)`；
- `boxsize=lengths` 告诉 KD-tree 这是周期盒，跨边界的邻居也会被找到；
- `query_ball_point(..., r_cut)` 返回每个中心在 `r_cut` 内的邻居 index 列表；
- `workers=-1` 用多核；`return_sorted=False` 省掉排序。

```python
counts = np.fromiter((len(x) for x in neighbours), dtype=np.int64, count=n)
ci = np.repeat(np.arange(n), counts)
cj = np.concatenate([np.asarray(x, dtype=np.int64) for x in neighbours])
```

把“列表的列表”压平成两个等长数组：
- 第 i 个中心有 `counts[i]` 个邻居；
- `ci` 里 i 重复 `counts[i]` 次；
- `cj` 是对应的邻居 index。

```python
keep = ci < cj
ci, cj = ci[keep], cj[keep]
```

每个 pair 在 i 和 j 两个中心各出现一次；只留 `i<j` 一次，避免重复计算。

```python
d = positions[cj] - positions[ci]
d -= lengths * np.round(d / lengths)
return ci, cj, d
```

注意：tree 是用 `wrapped` 建的，但 `d` 用原始 `positions` 算；最后一行做最小镜像：
`d = d - L * round(d/L)`，把每个分量折到 `[-L/2, L/2)`。

### 4.2 三斜盒 `_pairs_triclinic()`（第 173–225 行）

```python
inv_box = np.linalg.inv(box)
frac = positions @ inv_box
frac -= np.floor(frac)
```

把笛卡尔坐标转成分数坐标 `f = r @ box⁻¹`，再折到 `[0,1)`。  
（约定：`box` 的行是 a、b、c，`r = f @ box`。）

```python
shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
```

27 个 shift：`(-1,-1,-1) ... (1,1,1)`。

```python
images = (frac[None, :, :] + shifts[:, None, :]) @ box
flat_images = images.reshape(-1, 3)
image_ids = np.tile(np.arange(n), len(shifts))
centres = frac @ box
```

- `images` 形状 `(27, n, 3)`：每个原子在 27 个相邻晶胞里的镜像；
- `flat_images` 压成 `(27n, 3)`，给 KD-tree；
- `image_ids[k] = k % n`：第 k 个镜像对应哪个原始原子；
- `centres` 是折回主盒后的中心坐标。

```python
tree = cKDTree(flat_images)
neighbours = tree.query_ball_point(centres, r_cut, workers=-1, return_sorted=False)
```

这次 KD-tree 不需要 `boxsize`，因为 27 个镜像已经把周期邻居显式摆出来了。

```python
counts = ...
ci = np.repeat(np.arange(n), counts)
ck = np.concatenate([...])
cj = image_ids[ck]
d = flat_images[ck] - centres[ci]
r2 = np.einsum("ij,ij->i", d, d)
```

- `ck` 是镜像点 index；`cj` 还原成原始原子 index；
- `d` 是“中心 → 镜像邻居”的位移；
- `r2` 是每对的距离平方。

```python
keep = (r2 < r_cut**2) & (cj != ci) & (ci < cj)
```

只保留截断内、非自身、`i<j` 的 pair。

```python
key = ci * n + cj
order = np.lexsort((r2, key))
key_sorted = key[order]
first = np.ones(len(order), dtype=bool)
first[1:] = key_sorted[1:] != key_sorted[:-1]
selected = order[first]
```

同一个 `(i,j)` 可能通过多个镜像出现（尤其盒子和 `r_cut` 同量级时）。这里按 `(i,j)` 分组，组内按 `r2` 排序，只留距离最近的那个镜像。

> 为什么 27 个就够：如果 `r_cut` 小于最小垂直盒宽，任何在截断内的邻居，其最小镜像的分数坐标差都在 `[-1,1]` 内，27 个 shift 覆盖了所有可能。

### 4.3 暴力回退 `_pairs_bruteforce()`（第 228–259 行）

```python
for start in range(0, n, chunk):
```

按中心分块，避免一次分配 `N×N×3` 的大数组。每块：

```python
d = positions[None, :, :] - positions[start:stop, None, :]
frac = d @ inv_box
frac -= np.round(frac)
d = frac @ box
r2 = ...
ii, jj = np.nonzero((r2 < r_cut**2) & (r2 > 1e-12))
```

对块内每个中心，和所有原子算位移，做最小镜像，筛出截断内的 pair。`r2 > 1e-12` 排除自身。

```python
ii = ii + start
keep = ii < jj
...
d_parts.append(d[ii - start, jj])
```

把块内局部 index 加回全局 index，只留 `i<j`，并取出对应的 `d`。最后拼接各块。

这个函数主要给 `neighbor_method='bruteforce'` 和测试用；生产上正交/三斜路径更快。

---

## 5. 核心 kernel `compute_environment_similarity()`（第 262–399 行）

### 5.1 输入校验（第 304–319 行）

```python
pos = np.asarray(positions, dtype=np.float64)
box = np.asarray(box_vectors, dtype=np.float64)
```

统一转成 float64，避免后续广播时出现 float32/对象数组。

```python
if pos.ndim != 2 or pos.shape[1] != 3: ...
if box.shape != (3, 3): ...
if len(pos) == 0: ...
if sigma <= 0: ...
if cutoff < 0: ...
if lam <= 0: ...
if normalization not in (...): ...
```

参数检查，错误信息直接指出哪个参数不对。

### 5.2 参考环境与归一化（第 321–345 行）

```python
refs = [np.asarray(ref, dtype=np.float64) for ref in references]
```

`references` 是 `(n_m,3)` 的位移矢量列表，单位 nm。

```python
sizes = np.array([len(ref) for ref in refs], dtype=np.float64)
sizes_equal = bool(np.all(sizes == sizes[0]))
```

`n_m` 是每个参考环境的原子数。`sizes_equal` 是“所有模板一样大”的快速判断。

```python
if sizes_equal:
    norm_div = np.full(len(refs), sizes[0])
```

**常见情况**：所有模板一样大。这时 `per_reference` 和 `plumed` 完全一样，归一化因子就是同一个 `n`；不需要任何 warning。

```python
elif normalization == "plumed":
    warnings.warn(...)
    norm_div = np.full(len(refs), sizes[0])
```

模板大小不同、又要求 `plumed` 模式：PLUMED 会把所有模板都除以第一个模板的 `n0`，这不是“每个模板自归一化”，所以这里 warning，提示用 `per_reference`。

```python
else:
    norm_div = sizes
```

默认 `per_reference`：每个模板除以自己的 `n_l`。这是论文多模板公式的自洽做法，模板大小不同时也正确，因此**不 warning**。

```python
inv_norm = 1.0 / norm_div
```

后面用乘法 `* inv_norm[m]`，比每对都做除法略快。

### 5.3 截断与邻居方法（第 347–374 行）

```python
max_ref_norm = max(|r| for ref in refs for r in ref)
r_cut = max_ref_norm + cutoff * sigma
```

PLUMED 的 `CUTOFF` 是“在最大参考距离之外再看几个 σ”。默认 `cutoff=3`，σ=0.05 nm，所以 Ih 的 `r_cut≈0.43 nm`。

```python
method = neighbor_method
if method == "auto":
    if _is_orthorhombic(box):       method = "orthorhombic"
    elif r_cut < min(_box_widths(box)): method = "triclinic"
    else:                            method = "bruteforce" + warning
```

自动选择：
- 正交 → KD-tree + `boxsize`；
- 三斜且截断小于最小盒宽 → 27 镜像；
- 否则 → 分块 `O(N²)`。

### 5.4 kernel 本体（第 376–399 行）

```python
n = len(pos)
inv4s2 = 1.0 / (4.0 * sigma * sigma)
r2 = np.einsum("ij,ij->i", d, d)
values = np.zeros((len(refs), n))
```

- `n` 是中心数；
- `1/(4σ²)` 是高斯核的系数；
- `r2` 是每对 pair 的距离平方；
- `values[m, i]` 是第 i 个中心在第 m 个参考下的 kernel。

```python
for m, ref in enumerate(refs):
    ref_norm2 = np.einsum("ij,ij->i", ref, ref)
    dot = d @ ref.T
```

`dot[p,l] = d_p · r_l`。  
`d2 = |d - r|² = |d|² + |r|² - 2 d·r`，所以不需要显式做 `(P,n,3)` 的减法。

```python
d2_forward  = r2[:,None] + ref_norm2[None,:] - 2.0 * dot
d2_backward = r2[:,None] + ref_norm2[None,:] + 2.0 * dot
```

这是整个代码最关键的地方：

- 对 pair `(i,j)`，`d = r_j - r_i`；
- 中心 i 的环境看到 `r_j - r_i = d` → `d2_forward`；
- 中心 j 的环境看到 `r_i - r_j = -d` → `d2_backward`。

这个 CV 使用矢量分量，**破坏旋转对称性**，所以两个方向的 kernel 不一样。只算 `d2_forward` 然后同时加到 i 和 j，会让 Ih 冰的 `env_mean` 从 0.88 掉到 0.63。

```python
np.maximum(d2_forward, 0.0, out=d2_forward)
np.maximum(d2_backward, 0.0, out=d2_backward)
```

浮点误差可能让 `|d-r|²` 出现 `-1e-16`；截到 0，避免 `exp(+tiny)` 或数值噪声。

```python
kernel_forward  = np.exp(-d2_forward  * inv4s2).sum(axis=1) * inv_norm[m]
kernel_backward = np.exp(-d2_backward * inv4s2).sum(axis=1) * inv_norm[m]
```

对每个 pair：
- 对参考环境的所有原子 l 求 `exp(-|d-r_l|²/(4σ²))`；
- 沿 `axis=1` 求和；
- 乘 `1/n` 归一化。

结果 shape `(P,)`：每个 pair 对中心 i / j 的贡献。

```python
values[m]  = np.bincount(ci, weights=kernel_forward,  minlength=n)
values[m] += np.bincount(cj, weights=kernel_backward, minlength=n)
```

`bincount` 把“每个 pair 的贡献”按中心 index 累加：
- `ci` 是中心 i 的 index → 加 forward；
- `cj` 是中心 j 的 index → 加 backward；
- `minlength=n` 保证没有邻居的中心也得到 0。

```python
if len(refs) == 1:
    return values[0]

vmax = values.max(axis=0)
return vmax + np.log(np.exp(lam * (values - vmax)).sum(axis=0)) / lam
```

单个参考直接返回。多个参考做数值稳定的 softmax：
`(1/λ) log Σ exp(λ v_m)`。先减 `vmax`，避免 `exp(100*1)` 这种大数。

---

## 6. `more_than_count()`（第 402–439 行）

PLUMED 的 `MORE_THAN={RATIONAL R_0=r0 NN=nn MM=mm}` 的软计数：

```python
if mm == 0:
    mm = 2 * nn
```

PLUMED 约定：`MM=0` 等价于 `MM=2*NN`。

```python
r_max = (1e-5) ** (1.0 / (nn - mm))
r = values / r0
```

`r = k/R0`；`r_max` 是 switching function 截断到 `1e-5` 的位置。  
NN=8、MM=16 时 `r_max = 10^(5/8) ≈ 4.217`。

```python
s = np.zeros_like(r)
mask = r < r_max
rr = r[mask]
```

`r >= r_max` 时 `s=0`，所以 `1-s=1`。

```python
near_one = np.abs(rr - 1.0) < 1e-12
result[near_one] = nn / mm
result[~near_one] = (1.0 - rr[~near_one]**nn) / (1.0 - rr[~near_one]**mm)
```

`s(r) = (1-r^NN)/(1-r^MM)`。  
`r=1` 时是 0/0，取极限 `NN/MM`（8/16 → 0.5）。

```python
return float(np.sum(1.0 - s))
```

`n_ice = Σ [1 - s(k_i/R0)]`。这就是相图工作流里的 `env.morethan`。

**注意**：`n_ice` 不是 `sum(k > R0)`。一个正好在 `R0` 的分子贡献 0.5；略高于 `R0` 的分子也小于 1，直到 `r_max` 才完全变成 1。所以纯冰的 `n_ice` 约是 0.91N。

---

## 7. `EnvsimReporter` 类

### 7.1 docstring（第 442–484 行）

逐个参数说明，和 README 的参数表一致。

### 7.2 `__init__()`（第 486–596 行）

参数校验：

```python
if int(interval) <= 0: ...
if sigma <= 0: ...
if cutoff < 0: ...
if lam <= 0: ...
if r0 is not None and r0 <= 0: ...
if mm == 0: mm = 2*nn
if nn <= 0 or mm <= 0 or nn == mm: ...
if not isinstance(time_unit, Unit): ...
```

`time_unit` 现在是 `openmm.unit.Unit`，默认 `nanosecond`；传 `picosecond` 就输出 ps。不再用字符串查表。

```python
if isinstance(references, (str, os.PathLike)):
    references = [references]
```

允许 `references='ref1.pdb'` 或 `references=['ref1.pdb', ...]`。

```python
self._refs = [read_reference_pdb(path) * _ANGSTROM_TO_NM for path in references]
```

读参考 PDB（Å）并转 nm，只做一次。

`group`：

```python
if group is None:
    self._group = None
else:
    self._group = np.asarray(group, dtype=np.int64)
    if self._group.ndim != 1 or len(self._group) == 0: ...
    if np.any(self._group < 0): ...
```

传了 `group` 就用它；否则后面自动选水 O。

```python
self._ids = None
self.env = None
self.env_mean = None
self.n_ice = None
```

`_ids` 懒加载；`env/env_mean/n_ice` 是 report 后给主循环读的最新值。

主文件：

```python
if file is not None:
    self._out = open(file, "a" if self.append else "w")
    columns = ["#Step", f"Time({self.time_unit.get_symbol()})",
               "Temperature(K)", "env_mean"]
    if self.r0 is not None:
        columns.append("n_ice")
    self._out.write("\t".join(columns) + "\n")
```

- `time_unit.get_symbol()` 给出 `ns` 或 `ps`；
- `r0=None` 时列里没有 `n_ice`；
- 表头只在新文件写；`append=True` 时沿用已有文件。

逐原子文件：

```python
if per_atom_file is False:
    atom_path = None
elif per_atom_file is None:
    if self.r0 is None:
        atom_path = <file>_env.dat
    else:
        atom_path = None
else:
    atom_path = per_atom_file
```

- `r0=None`（标定模式）→ 自动写 `<file>_env.dat`；
- `r0` 给定 → 默认只写 summary；
- `per_atom_file=False` → 强制不写；
- 传路径 → 写指定文件。

```python
if os.path.abspath(atom_path) == os.path.abspath(file):
    raise ValueError(...)
```

防止主文件和逐原子文件写成同一个路径。

### 7.3 `describeNextReport()`（第 601–604 行）

```python
steps = self.interval - simulation.currentStep % self.interval
return {"steps": steps, "periodic": True, "include": ["positions"]}
```

OpenMM 8.x 的 dict 形式：
- `steps`：距离下一次 report 还有几步；
- `periodic=True`：positions 折回主盒；
- `include=["positions"]`：只要坐标，不要 forces/energy/velocities。

### 7.4 `report()`（第 606–660 行）

第一次调用：

```python
if self._ids is None:
    self._ids = self._select_ids(simulation.topology)
    if self._atom_out is not None and not self.append:
        header = "time " + " ".join(f"env.{i+1}" for i in range(len(self._ids)))
        self._atom_out.write(header + "\n")
```

选择水 O，并写逐原子文件表头 `time env.1 ... env.N`。表头只在非 append 时写。

取坐标和盒子：

```python
positions = np.asarray(
    state.getPositions(asNumpy=True).value_in_unit(nanometer)
)[self._ids]
box_vectors = np.asarray(
    state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometer)
)
```

- `state` 里已经有 positions（`describeNextReport` 请求的）；
- 单位 nm；
- 只保留选中的 O，减少后续计算量。

计算：

```python
env = compute_environment_similarity(
    positions, box_vectors, self._refs,
    sigma=self.sigma, cutoff=self.cutoff, lam=self.lam,
    normalization=self.normalization,
)
```

存属性：

```python
self.env = env
self.env_mean = float(env.mean())
n_ice = None if self.r0 is None else more_than_count(env, self.r0, self.nn, self.mm)
```

写主文件：

```python
time_out = state.getTime().value_in_unit(self.time_unit)
temperature = self._get_temperature(simulation)
parts = [str(simulation.currentStep), f"{time_out:.6f}",
         "\\" if temperature is None else f"{temperature:.2f}",
         f"{self.env_mean:.6f}"]
if n_ice is not None:
    parts.append(f"{n_ice:.6f}")
self._out.write("\t".join(parts) + "\n")
```

时间单位由 `self.time_unit` 决定；温度拿不到时写 `\`；`n_ice` 只在 `r0` 给定时写。

写逐原子文件：

```python
self._atom_out.write(
    f"{time_out:.6f} " + " ".join(f"{v:.6f}" for v in env) + "\n"
)
```

每行 `time env.1 ... env.N`，空格分隔。

### 7.5 `close()` / `__del__()`（第 662–678 行）

```python
for attr in ("_out", "_atom_out"):
    handle = getattr(self, attr, None)
    if handle is not None:
        handle.flush()
        handle.close()
        setattr(self, attr, None)
```

关闭两个文件；`__del__` 兜底调用。关闭后把属性置 None，避免重复关闭。

### 7.6 `_select_ids()`（第 683–703 行）

```python
if self._group is not None:
    return self._group.copy()
```

用户显式给了 group，直接用。

```python
for atom in topology.atoms():
    if atom.residue.name not in self.water_names:
        continue
    element = getattr(atom, "element", None)
    if element is not None:
        if element.symbol != "O":
            continue
    elif atom.name not in self._oxygen_names:
        continue
    ids.append(atom.index)
```

自动选：
- residue name 在 `water_names` 里；
- 有 element 时必须是 O；
- 没有 element 时退回原子名（O/OW/OH2/...）；
- 收集 0-based index。

```python
if not ids:
    raise ValueError("no water oxygen atoms found; ...")
```

一个都没选到时明确报错，避免静默写 0。

### 7.7 `_get_temperature()`（第 705–719 行）

```python
getter = getattr(simulation.integrator, "getTemperature", None)
if getter is not None:
    return float(getter().value_in_unit(kelvin))
```

Langevin 等积分器有 `getTemperature()`，直接读。

```python
for name in simulation.context.getParameters():
    if "Temperature" in name:
        return float(simulation.context.getParameter(name))
```

没有 `getTemperature()` 时，退而求其次在 context parameters 里找名字含 `Temperature` 的参数。都找不到就返回 `None`，主文件写 `\`。

---

## 8. 几个容易看不懂的点

### 8.1 为什么 `values` 是 `(M, N)`？

`M` 是参考环境个数，`N` 是中心原子数。  
`values[m,i]` 是第 i 个中心在第 m 个参考下的 kernel。最后沿 `M` 做 softmax 得到 `env[i]`。

### 8.2 为什么每个 pair 要算 forward 和 backward？

因为 CV 用的是矢量分量 `(x,y,z)`，不是旋转不变量。  
对 pair `(i,j)`，中心 i 看 `r_j-r_i`，中心 j 看 `r_i-r_j = -(r_j-r_i)`。  
`|d-r_l|²` 和 `|-d-r_l|²` 不一样，所以必须分别算。

### 8.3 为什么用 `bincount` 而不是循环每个中心？

`bincount` 是 C 实现的按 index 累加：
`bincount(ci, weights=kernel_forward)` 等价于
`for p in range(P): values[ci[p]] += kernel_forward[p]`，但快得多。

### 8.4 为什么 `minlength=n`？

有些中心在截断内没有邻居（比如刚汽化的分子）。`bincount` 默认返回长度 `max(ci)+1`，不给 `minlength` 会短一截，导致 `values[m]` 形状不对。

### 8.5 为什么 `np.maximum(d2, 0)`？

`|d-r|²` 由 `|d|²+|r|²-2d·r` 算出来，浮点误差可能给 `-1e-16`。截到 0 避免 `exp` 出现不必要的大数/噪声。

### 8.6 为什么 `r_cut = max|r0| + 3σ`？

PLUMED 的 `CUTOFF` 定义：在最大参考距离之外再看 `CUTOFF*SIGMA`。  
`CUTOFF` 默认 3，`SIGMA=0.05 nm` → 多 0.15 nm。Ih 的 `max|r0|≈0.28 nm`，所以 `r_cut≈0.43 nm`。

### 8.7 默认且所有参考等大时，归一化还能更简单吗？

可以。代码里已经有一个快速分支：

```python
sizes_equal = np.all(sizes == sizes[0])
if sizes_equal:
    norm_div = np.full(len(refs), sizes[0])
```

这时 `per_reference` 和 `plumed` 完全一样，归一化就是一个常数 `1/n`；`inv_norm` 也是常数数组。  
只有当模板大小不同时，才需要区分“每个模板除自己的 `n_l`”和“全部除第一个 `n0`”。

### 8.8 为什么参考 PDB 不用 `openmm.app.PDBFile`？

`ref1.pdb` 的 4 个原子共用/重复 serial，OpenMM 的 PDB 读取器会按 serial 去重，读成 1 个原子。  
这里按固定列直接读，保留全部 `ATOM` 行。

---

## 9. 可以怎样扩展

- 多物种（`SPECIESA/SPECIESB`、`ATOM_NAMES_FILE`）：需要把 `refs` 按原子名分组，并在 `d2` 里加 species mask。
- 更大的体系：把邻居搜索换成纯 NumPy cell list，或者把 `d2` 的 `(P,n)` 计算改成 float32。
- 异步：把 `compute_environment_similarity` 放到后台线程，和下一步 GPU 计算重叠；当前实现是同步的，逻辑更简单。
- 输出更多统计：`reporter.env` 已经是逐原子数组，主循环可以自行算分位数、最大团簇等。
