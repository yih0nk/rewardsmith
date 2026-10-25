# Devpost submission

Hackathon page: https://cal-hacks-13-0.devpost.com. **Deadline: Sun Oct 25, 9:00 AM PDT, hard.** Submit by 8:30.

## What Devpost requires (from the 13.0 page, checked Oct 25 at 1 AM)

- [ ] Be at the venue (Palace of Fine Arts) and checked in by **8:00 AM**. You must be checked in as a hacker to submit.
- [ ] Submit by **9:00 AM PT**, "with your teammates": every teammate is added to the Devpost project.
- [ ] **An image of the project** (gallery image) and **the GitHub repository** for any code. **yih0nk/rewardsmith is private, so make it public before submitting.**
- [ ] All work done during the hackathon. Everything in the repo is from Oct 23–25. Don't link or mention any earlier project.
- [ ] Team of 2–4, college students, 18+.
- [ ] Judging: a **3-minute pitch + 2 minutes of Q&A**, with a pre-recorded or live demo.
- [ ] Devpost's judging criteria ask for "a clear, meaningful **video demo** on devpost for sponsors evaluating asynchronously". Add the 2-minute video link (`demo-video.md`) to the submission.

Judging criteria (Devpost): **Application** (who it helps, feasible adoption), **Intentionality** (purposeful design for its users), **Functionality** (live demo runs smoothly, plus the video), **Creativity**, **Technical Complexity**.

---

## Project name

**Rewardsmith**

## Tagline (≤ 60 characters on Devpost; this one is 59)

Describe a robot behavior in English. Get a trained policy.

## Inspiration

Teaching a robot something new is still a project. One path is to hand-write a reward function and spend days tuning weights until the policy stops exploiting it. The other is to collect thousands of demonstrations for a generalist model. We wanted "try a new behavior" to be a sentence you type, with an honest answer about whether it worked, in the time it takes to get coffee.

## What it does

You type a behavior, like "push the ball into the goal", "hop with all four feet", or "walk to the blue box". Then:

1. **Claude builds the scene** (objects, a goal zone, terrain) and **writes a pass/fail success check** before any training, for example "the ball ends inside the goal in ≥ 80% of episodes, falls in ≤ 10%". The check is fixed from then on.
2. **Claude writes reward functions as JAX code.** Each one must compile and survive a short simulated rollout. If it crashes, Claude reads the error and fixes it.
3. A simulated **Unitree Go1** trains with **PPO in MuJoCo Playground** on an RTX 4090. It learns to **steer a walker we pretrained**, so each task only has to learn the task, not how to walk.
4. While it trains, a **live coach** (Claude Haiku) comments on every checkpoint ("1.3M steps: walks at 0.5 m/s but the ball is still 1.4 m short of the goal").
5. Every policy runs **32 fresh randomized episodes** against the success check. A **critic** (Claude) watches frames of the rollout video, and code caps its score when the check fails.
6. It's a **conversation**: follow up on any candidate ("get behind the ball first", or just "keep going"). The follow-up continues from that trained policy instead of starting over.

**Instant** mode gives you one reward and a first video in about 2.5 minutes. **Effort** mode gives you four rewards side by side with the coach in about 5 minutes. You can pick Opus, Sonnet or Haiku per message.

## How we built it

- **Simulation:** MuJoCo Playground's Go1 environment on MJX with the Warp backend, and brax PPO, on one Runpod RTX 4090. A pretrained joystick walker (213M steps in 5.5 minutes) is frozen. Each task's policy outputs smoothed velocity commands to it ("steer"). Pose tasks fine-tune the walker instead.
- **Scenes:** a JSON scene spec (balls, boxes, zones, rough terrain) that we compile into MJCF ourselves. Claude never writes model XML. Objects are placed per episode, with domain randomization on the robot.
- **Success checks:** a vocabulary of per-episode outcomes (`push:ball:goal`, `reach:box`, `climb`, `sequence:…`) plus pose metrics computed from forward kinematics. They're measured over 32 episodes in the same batched rollout that renders the video.
- **Agents:** Claude Opus 5.5 writes and edits rewards and judges results. Sonnet 5.5 writes the scene and the check (and runs Instant mode). Haiku 5.5 is the live coach and reads follow-ups ("does this note change the scene?"). Ideas stream to the board while Claude is still writing.
- **Backend:** a FastAPI server and a worker process share SQLite. The worker keeps warm training processes, so the compile gate and PPO start while Claude is still writing.
- **Board:** React + Vite + TypeScript, in a black-and-green terminal design. Conversations, a live step strip with server-side stage timestamps, the success-check table, the coach feed, reward diffs between rounds, and pause/resume/cancel.

## Challenges we ran into

- **A test that proved nothing.** "Move the ball to its destination" passed its success check with a critic score of 3, because the check only asked "didn't fall", and our metrics measured the robot reaching the ball, not the ball reaching the goal. We added task metrics, a rule that every check needs a task-specific clause, and a hard cap on the critic score when a task clause fails.
- **The episode clock.** Push policies flipped in 100% of 20-second evaluation episodes, while the 10-second videos looked fine. Training resets every environment after about 6.4 s, but the policy could see an episode-time input, so past 6.4 s it was out of distribution. Removing that input took flips to 3%.
- **Learning to walk every time.** Training every task from random weights spent most of the budget on walking and produced a crouched shuffle. Building on a pretrained walker fixed that. Fine-tuning it made policies stand still (0% ball reached), but steering it reached the ball in 100% of episodes for 3 of 4 rewards.
- **Latency.** Four PPO compiles share about 10 CPU cores on the pod. We cut the first video from 339 s to 280 s (warm processes, batched evaluation, no double compile), then split the product into Instant and Effort, so a first video comes in about 2.5 minutes.
- **Physics surprises:** Playground damps every free joint, including our ball. MJX has no box–cylinder collisions. Warp floods the logs with solver warnings until you silence them.

## Accomplishments that we're proud of

- On "make robot move ball to a target destination", our first version scored **2/10**: reach-and-push worked in **15.6%** of episodes and the robot fell in **37.5%**. Now Claude's own check passes, the push succeeds in **72–100%** of episodes, and falls are **0–3%**. On the live board the same prompt scored **8/10**, with **96.9%** reach-and-push and **0%** falls.
- On "push the ball into the goal zone", the ball ended in the zone in **88–100%** of episodes, up from 0–28%.
- A follow-up, "get behind the ball first", took ball-reached-goal from **78% to 100%** on our test server, continuing from the trained policy.
- It **says no honestly.** Hind-leg stands still fail, and the board shows them failing (best 3/10) instead of passing a crouch. In our scene-builder PR, Claude classified 30 of 32 judge-style prompts correctly and refused all 9 impossible ones (e.g. "pick up the ball with your mouth" → "the Go1 has no mouth or gripper").
- A live coach that watches every checkpoint for about **a cent** per run.

## What we learned

- Measure per episode, not per video. Every big quality jump came from a metric that disagreed with a good-looking clip.
- Make the LLM commit to the test before it sees results, and let code, not the LLM, have the last word on the score.
- Give the learner a body that already walks. The reward then only has to describe the task.
- Early stopping on scene tasks hurts: candidates that fall at 5M steps are often still learning. We made the coach advise-only by default.

## What's next

**Same loop, any body.** The scene compiler, success checks, reward writer and critic don't depend on the dog. Next are a humanoid and an arm from MuJoCo Playground, then stairs/ramps/gaps (built, in review), Claude-written success checks as gated code (in review), one-click policy export (in review), and sim-to-real hardening with robustness scores per policy and an ONNX export.

## Built with

`claude` `anthropic-api` `claude-opus` `claude-sonnet` `claude-haiku` `python` `jax` `brax` `mujoco` `mujoco-playground` `mjx` `nvidia-warp` `ppo` `fastapi` `sqlite` `react` `typescript` `vite` `runpod` `nvidia-rtx-4090` `ffmpeg` `uv`

## Links

- Try it: https://o95f0trj2xjkjb-8000.proxy.runpod.net (only up while the pod runs; say so in the submission)
- Code: https://github.com/yih0nk/rewardsmith (**public by submission time**)
- Video: *(YouTube or Vimeo link, unlisted is fine; see `demo-video.md`)*

---

## Prize tracks: which to opt into

These come from https://hive.hackberkeley.org/prizes and the Devpost prize list (updated Oct 24, 3:13 PM). The Cal Hacks overall prizes and Hacker's Choice apply to every submission.

| Track | Opt in? | One-line justification |
|---|---|---|
| **[Anthropic] Claude for Public Benefit** ($5,000 Claude credits + 1 yr Max 20x) | **Yes** | Claude does every hard step (scene, test, reward code, coaching, judging), and we can show where it goes wrong and what guards it: code caps, honest failures, refusals. **Risk:** the criteria want "a named community" and a problem "you have lived, or talked to someone who has". Name a specific group in the write-up (e.g. a university robotics lab without an RL engineer) **only if a teammate actually talked to them.** |
| **Cal Hacks Most Technically Complex Hack** | **Yes** | MJX/Warp PPO, a scene compiler, per-episode outcome checks, a pretrained-walker steering layer and a multi-agent loop on one GPU. This is our strongest track. |
| **Cal Hacks Most Creative Hack** | Yes | "A sentence in, a trained robot out, and it argues with itself about whether it worked" is an unusual framing. It costs nothing to opt in. |
| **[MLH] Commit Fellowship** | Optional | Only if the team wants a 3-week founder sprint. It needs the supplemental form after submitting. |
| [Factory] Best Use of the Droid SDK ($2,000) | **No, unless** the live worker runs `REWARDSMITH_EDITOR=droid` | The README documents a Droid editor, but the live board uses Claude's editor ("Claude's edit to the reward"). Factory scores "only in the README" poorly. Switching the live worker the morning of the pitch is not worth the risk. |
| [Microsoft] Built for the Real World | No | It wants a production-minded agent in finance, productivity or healthcare, with a path to production. A sim-only robot trainer is a weak fit. |
| Cal Hacks Best Hardware Hack | No | Simulation only. Claiming hardware would be dishonest. |
| Everything else (Convex, Fetch.ai, Hyphenate, Visa, Apify, Cartesia, Bright Data, Spacetime, Mana, Creao, …) | No | We don't use their products. |
