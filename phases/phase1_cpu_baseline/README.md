# Phase 1 — CPU Baseline / CPU 基准

**Goal / 目标:** Run **10,000 particles** entirely on the CPU, draw them with OpenGL, and record how
fast it goes. This is your *reference benchmark* — every later phase is measured against the numbers
you see here.

> 用纯 CPU 跑 1 万个粒子，用 OpenGL 画出来，并记录速度。这就是“参照基准”——后面每个阶段都拿它来对比。

---

## 1. The big picture / 全局图

A particle system is just a big array of tiny objects, each with a position, a velocity, and a
lifetime. Every frame you do the same three things:

> 粒子系统就是一个大数组，每个元素是一个小粒子，有位置、速度、寿命。每一帧都做同样的三件事：

```
   for each particle:           对每个粒子：
       move it a little           移动一点点
       age it a little            老化一点点
       if it died, restart it     如果死了，就重生
```

That `for each particle` loop is the heart of everything. Notice that **each particle is completely
independent** — particle #5 never needs to look at particle #6. Hold onto that idea: it is exactly
*why* a GPU can later run all 10,000 (or 1,000,000) of them at the same time.

> 这个“对每个粒子”的循环是核心。关键点：**每个粒子完全独立**——5 号粒子不需要看 6 号粒子。
> 记住这一点，这正是 GPU 之后能同时处理 1 万甚至 100 万个粒子的原因。

---

## 2. What's already built for you / 框架已经写好的部分

| File | What it does | Edit it? |
|------|--------------|----------|
| `src/main.cpp` | Opens the window, runs the loop, measures FPS + update time | ❌ no |
| `src/renderer.cpp/.h` | Draws particles as colored OpenGL points | ❌ no |
| `src/particle_system.h` | The simulation. `reset()`, `respawn()`, `to_vertices()` are done | ✅ **yes — one function** |
| `include/particles.h` | The `Particle` struct and `SimParams` | ❌ no (read it) |

You only write **one function**: `ParticleSystem::update(float dt)`.

> 你只需要写**一个函数**：`particle_system.h` 里的 `update(float dt)`。其它全部已就绪。

---

## 3. Your task / 你的任务

Open [`src/particle_system.h`](src/particle_system.h) and find the `── TODO ──` block inside
`update(float dt)`. Write a loop over all `params_.n` particles. For each particle `i`:

> 打开 `src/particle_system.h`，找到 `update` 里的 `── TODO ──`，写一个遍历所有粒子的循环。对每个粒子 `i`：

| Step | 中文 | Code shape |
|------|------|------------|
| 1. Gravity | 重力让速度往下掉 | `particles_[i].vy -= params_.gravity * dt;` |
| 2. Move | 用速度更新位置 | `particles_[i].x += particles_[i].vx * dt;` (and `y`) |
| 3. Age | 寿命减少 | `particles_[i].life -= dt;` |
| 4. Recycle | 死了就重生 | `if (particles_[i].life <= 0.0f) respawn(i);` |

That's the entire physics for Phase 1. Keep each iteration self-contained.

`dt` (delta-time) is the real number of seconds since the last frame, measured by `main.cpp`. Using
it instead of a fixed step means the animation runs at the same *speed* whether your machine does
60 or 6000 FPS.

> `dt` 是距离上一帧的真实秒数。用它而不是固定步长，动画速度就不会随帧率变化。

---

## 4. Build & run / 编译运行

From the project root:

```bash
cmake -B build -S .
cmake --build build --parallel
.\build\phases\phase1_cpu_baseline\Release\phase1.exe
```

Or use the helper script (configures, builds, and runs in one go):

```bash
.\build.bat run
```

> 第一次编译会自动下载并编译 GLFW，需要联网，耐心等一会儿。之后就很快了。

**What you should see:** a dark window with a fountain of warm-colored points arcing up from the
bottom center and falling under gravity, looping forever. The title bar shows
`10000 particles | <FPS> | update <ms>`.

> 你会看到：底部中央喷出一束暖色粒子，受重力下落，循环不停。标题栏显示粒子数、FPS、更新耗时。

---

## 5. Record your baseline / 记录基准

Write these down — you'll compare against them in Phase 2:

> 把这两个数字记下来，第二阶段（CUDA）要对比：

- **FPS** at 10,000 particles: ________
- **`update` time** (ms per frame): ________

Curious? Bump `kNumParticles` in `main.cpp` to `100000`, rebuild, and watch the `update` time climb.
The CPU does every particle one after another, so the cost grows linearly. That wall is what the GPU
is about to smash through.

> 想试试？把 `main.cpp` 里的 `kNumParticles` 改成 100000 再编译，看 `update` 时间飙升。
> CPU 一个一个算，开销线性增长。这堵墙，正是 GPU 接下来要砸穿的。

---

## 6. Checklist / 检查清单

- [ ] The fountain animates smoothly and loops (particles respawn, never disappear forever).
- [ ] The title bar shows a stable FPS and an `update` time.
- [ ] You recorded your baseline FPS and update-time numbers.

When all three are done, you've cleared **Level 1**. Before Phase 2, try **Level 2** below — more
code to write, more C++ to learn.

> 三项都完成 = 通过**第一关**。进入 Phase 2 之前,先挑战下面的**第二关**:多写点代码,多学点 C++。

---

## Level 2 / 第二关 — Bounce off the walls / 撞墙反弹

Right now particles that fly off the edge just keep going until they die. Let's make them **bounce**
off the four walls instead, like a ball in a box. This is more practice with `if` and a new C++
idea: the **unary minus** (flipping a sign).

> 现在粒子飞出边界就一去不回。我们让它像盒子里的弹球一样**撞墙反弹**。这一关多练 `if`,
> 还会学一个新 C++ 概念:**取负号**(把正负翻过来)。

### The physics / 物理

A wall sits at each edge: the world spans `-bound … +bound` on both x and y (`bound` is in
`SimParams`, value `1.0`). When a particle crosses a wall, two things happen:

> 四面墙在 `-bound` 到 `+bound`(`bound` 在 `SimParams` 里,值是 1.0)。粒子越过墙时做两件事:

1. **Flip the velocity** in that direction — moving right becomes moving left. In C++ you flip a
   sign with a minus in front: `vx = -vx;`  (this is the *unary minus* — it negates one value).
2. **Lose a little energy** — multiply by `restitution` (0.8) so each bounce is a bit weaker:
   `vx = -vx * params_.restitution;`

> ① **翻转速度**:向右变向左。C++ 里在前面加个负号就翻转:`vx = -vx;`(这就是"取负号")。
> ② **损失一点能量**:乘 `restitution`(0.8),每次反弹弱一点:`vx = -vx * params_.restitution;`

### New C++ concept: unary minus / 新概念:取负号

```cpp
float a = 5.0f;
float b = -a;     // b 现在是 -5.0  —— 前面一个减号,就是"取相反数"
```

You've seen `-` as subtraction (`a - b`). Here it has **one** operand and just flips the sign. A
velocity of `+0.3` (moving right) becomes `-0.3` (moving left) — a bounce.

> 你见过 `-` 当减法(`a - b`,两个数)。这里它只跟**一个**数,作用是翻正负号。
> 速度 `+0.3`(向右)变成 `-0.3`(向左),就是反弹。

### Your task / 你的任务

In `update()`, **after** the move step (after you update `x` and `y`), add four `if` checks — one
per wall. Here's the **right wall** done for you as the model:

> 在 `update()` 里,**移动之后**(更新完 `x`、`y` 之后)加四个 `if`,一面墙一个。右墙我做示范:

```cpp
// hit the RIGHT wall? (x went past +bound)
if (particles_[i].x > params_.bound) {
    particles_[i].x  = params_.bound;                          // put it back on the wall
    particles_[i].vx = -particles_[i].vx * params_.restitution; // bounce: flip + lose energy
}
```

Now you write the other three, following the same shape:

> 照着这个形状,你写另外三面:

| Wall / 墙 | Condition / 条件 | Which velocity flips / 翻哪个速度 |
|------|------|------|
| Right 右 (done 已示范) | `x > bound` | `vx` |
| Left 左 | `x < -bound` | `vx` |
| Top 上 | `y > bound` | `vy` |
| Bottom 下 | `y < -bound` | `vy` |

> 提示:左墙条件是 `x < -params_.bound`(注意这里 `-params_.bound` 又用到了取负号!)。
> 上下两面墙改的是 `y` 和 `vy`,逻辑和左右完全一样。

Build, run, and watch particles ping around inside the box, each bounce a little gentler until they
settle. When that works, you've cleared Level 2 — and you're ready for **Phase 2 — CUDA Migration**,
where this exact loop moves onto the GPU.

> 编译运行,看粒子在盒子里乒乓乱跳,一次比一次温柔,最后慢慢稳下来。成了就通过第二关——
> 接下来就是 **Phase 2:把这个循环搬上 GPU**。

---

## Hints / 提示

<details>
<summary>I get a black/empty window.</summary>

Did you write the `update()` loop? With an empty `update()`, every particle stays at the emitter
point and nothing moves. Also confirm the build picked up your edit (rebuild).
</details>

<details>
<summary>Particles fly away and never come back.</summary>

Make sure step 4 calls `respawn(i)` when `life <= 0`. Without it, particles age past zero, keep
drifting, and eventually leave the screen for good.
</details>

<details>
<summary>Everything falls instantly / looks frozen.</summary>

Check the order: apply gravity to `vy` *before* moving, and multiply by `dt` everywhere
(`* dt`). Forgetting `dt` makes the motion depend on frame rate and usually looks far too fast.
</details>

<details>
<summary>Show me the answer.</summary>

```cpp
void update(float dt) {
    for (int i = 0; i < params_.n; ++i) {
        particles_[i].vy   -= params_.gravity * dt;        // 1. gravity
        particles_[i].x    += particles_[i].vx * dt;       // 2. move
        particles_[i].y    += particles_[i].vy * dt;
        particles_[i].life -= dt;                          // 3. age
        if (particles_[i].life <= 0.0f) respawn(i);        // 4. recycle
    }
}
```

Try writing it yourself first — the muscle memory matters when this becomes a CUDA kernel in Phase 2.
</details>
