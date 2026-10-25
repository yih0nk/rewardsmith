# Judge Q&A: 2 minutes, crisp answers

Answer in 2–3 sentences, then stop. If a judge wants more, they'll ask. Every number here is in the **Numbers** table at the bottom, with its source. Don't quote a number that isn't there.

---

### "How do you know it actually did the task?"
Before any training, Claude writes a pass/fail test from your sentence, for example "the ball ends in the goal in at least 80% of episodes, falls in at most 10%". It can't change the test after results come in. Every trained policy then runs 32 fresh randomized episodes and the board shows Got vs Need for each clause. A second Claude also watches the video, and code caps its score at 4 if any task clause fails, so a pretty video can't outvote the measurement.

### "How is this different from Eureka?"
Eureka showed LLMs can write reward code. But each task comes with a human-made environment and a fitness score to select rewards against, and the evolutionary search runs many full trainings. We generate the scene and the success test from the sentence too, judge with an independent critic, and build on a pretrained walker, so one Effort run is about 5 minutes on one 4090. It's also conversational: you follow up on a candidate and it continues from that policy.

### "And Language to Rewards?"
L2R has an LLM fill in parameters of a reward template that a model-predictive controller optimizes in real time. That's fast, but it's bounded by the template and gives you a controller, not a learned policy. We write free-form reward code and train a neural policy you can export, then verify it with a test written before training.

### "Why not a generalist robot model like RT-2 or π0?"
They're trained on very large teleoperated demonstration sets and are strong at manipulation that resembles their data. A new dynamic skill, like hopping or pushing a ball into a goal, would need new demos. We need zero demos: the sentence becomes a reward and simulation does the practice. The two are complementary: our output could be training data or a skill for one of them.

### "Does it transfer to a real robot?"
Honestly, it's simulation only today; we haven't put it on a Go1. What helps: training randomizes the robot's physical parameters, and the export (open PR #50) produces a policy file plus a script that replays it outside our board. That replay matched the board's numbers for a steer policy, e.g. upright 0.999 and height 0.305 m on both. A balance-edge pose policy did not reproduce on a different physics backend, which is exactly the kind of gap sim-to-real would expose.

### "How long does a task take? What does it cost?"
Instant (one reward) took 2m 23s–2m 34s end to end on the live board. Effort (four rewards side by side with the live coach) took 4m 54s. About a minute of every run is the fixed PPO compile. GPU time is a few cents per run on a rented 4090, and the live coach is about one cent per Effort run (Haiku, about 3 s per look). We haven't metered the Opus/Sonnet writing and judging per run.

### "What doesn't work? What are the limits?"
One body, the Unitree Go1 quadruped, and no grasping. Hind-leg stands are unsolved: the robot rears up and falls within about a second, and our test fails it honestly; see the hind-leg run, best 3.0. The push gait is still a bit crouched and nose-down. Follow-ups don't always improve: on the live board, "stop the ball inside the goal" made it overshoot.

### "What about reward hacking?"
It's the thing we designed around. In our first version, "move the ball to the destination" passed its test because the test only checked "didn't fall". Now the test must contain a task-specific clause, the critic's score is capped at 4 when a task clause fails, and every result has a "Flag as reward hacking" button. In open PR #49, checks Claude writes as code are rejected if a robot standing still would pass them.

### "Why Claude?"
Three jobs need different things. Opus writes and edits reward code and reads physics errors to fix its own crashes. Sonnet builds the scene and the test quickly. Haiku is a cheap live coach that comments on every checkpoint during training. The critic reads video frames plus numbers, which needs a model that can look and do arithmetic. And you pick the model per message on the board.

### "Isn't the pretrained walker cheating?"
It's the opposite of cheating; it's the point. Learning to walk from scratch for every request wasted most of the budget, and the gait was a crouched shuffle. The walker trains once (213M steps in 5.5 minutes). Each task's policy then learns to steer it, so the policy learns the task, not how to stand up. For poses we fine-tune the walker instead, because steering can't lift a leg.

### "How did you find the bugs?"
By measuring per episode instead of trusting videos. The 10 s video hid that the push policy flipped in 100% of 20 s episodes. The cause was an episode-clock input the policy never saw past 6.4 s during training. Removing it took flips from 100% to 3%.

### "Is running LLM-written code safe?"
Each reward must compile and survive a short jitted rollout before training, and errors go back to Claude to fix (up to 3 tries). In open PR #49, success checks written as code also pass an AST guard (only jax/math imports, no file or eval access) and a dry run. Rewards don't have that AST guard yet; that's the next hardening step.

### "Who is this for?" (the Anthropic track asks this)
University robotics labs and students who have a simulator but no reward-engineering expertise, and roboticists who want to try ten behaviors before committing a month to one. *(Yihan: name the specific person or lab you talked to here. Don't invent one.)*

### "What did you build here vs. use off the shelf?"
Off the shelf: MuJoCo Playground's Go1 model, brax PPO, MuJoCo/Warp physics, and the Claude API. Everything else we built this weekend: the scene compiler, the success-check vocabulary and evaluator, the agent loop (writer, editor, critic, coach), the pretrained-walker steering, conversations and follow-ups, and the board.

### "What's next?"
Same loop, any body: the scene, test, reward and critic don't care that it's a dog. Next are a humanoid and an arm in MuJoCo Playground. Then hardening for transfer: domain-randomization scores per policy and an ONNX export.

---

## Numbers (sourced)

| Claim | Number | Source |
|---|---|---|
| Ball-to-destination, before | critic 2.0, reach+push 15.6%, falls 37.5% | #39 table / #47 (live request 6d9cfe59) |
| Ball-to-destination, after | critic 7/8/8/4, push 72–100%, falls 0–3%, check passed 3/4 | #39 "after (final)" / #47 |
| Same prompt on the live board | 8.0, reach+push 96.9%, falls 0% | live board `#9d26b109` turn 1 |
| "Push the ball into the goal zone" | ball in zone 0–28% → 88–100%; falls 100% → 0–9% | #39 table |
| Episode-clock bug | reference push flips 100% → 3%; reach 91% → 0% | #39 fixes |
| Follow-up "get behind the ball" | ball reached goal 78% → 100%; "wrong way" 6% → 0%, "fell short" 16% → 0% | chat-runs session on its test server, not in a PR body. Say "on our test server". |
| Follow-ups that got worse | 8.0 → 3.0; overshoot 72% and 88% | live board `#9d26b109` turns 2–3 |
| Latency, first video (K=4, one round) | 339 s → 280 s | latency PRs #30/#34/#35 (session notes) |
| Instant end to end | 2m 23s–2m 34s; earlier measure: first video 133 s, done 143 s | commit 2702f3f ("a 2m 23s run"), live `#be7cc83f` (2m 34s); chat-runs bench (133 s / 143 s, before the 3M-step cap) |
| Effort end to end | 4m 54s (one round of 4, with coach) | live `#96acf8a0` |
| Two Effort rounds (why Effort is now one) | 537 s, round 2 ~260 s, no better score | commit 507e4bd |
| PPO compile | ~1 min fixed per run (~65 s; ~90 s with 4 compiling on ~10 CPUs) | latency notes, modes.py |
| Pretrained walker | 213M steps in 330 s (5.5 min; 39 s compile), 0 falls in 64 randomized robots | #43 / docs/warm-start.md |
| Steer vs fine-tune on ball rewards | ball reached 100/100/100/0% steer vs 0% tune | #43 |
| Live coach | Haiku 5.5; median ~3 s per look; $0.007–0.012 per run | #41, #42, live `#96acf8a0` |
| Refusals (open #48) | 30/32 classified right; 9/9 impossible refused; 31/32 scenes passed the physics gate | #48 |
| Export (open #50) | steer policy replays on CPU: fwd 0.736 vs board 0.682 m/s, height 0.305 both | #50 |
| Hind legs | best 3.0 (live `#cca408b2`); policies rear up and fall in ~1 s | #39/#47, live board |
| Hop, all four feet | 8.0, check passed | live `#5b84e1b9` |
| Rough terrain | 0.45–0.92 m/s, critic 7–8, 0 of 4 fell | #39 |
| Scene physics gate (open #48) | 0.2–0.6 s per scene once warm | #48 |
| Tests | pytest 396 passed on main at #47 | #47 |
