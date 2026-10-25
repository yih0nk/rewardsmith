# Live demo script, click by click

Board: https://o95f0trj2xjkjb-8000.proxy.runpod.net (the pod's proxy URL; new task is `#new`, a conversation is `#<id>`).

**One rule:** the worker runs one request at a time, in order, on one RTX 4090. Anything you submit while another run is training shows "Queued, starts when “…” finishes". So during judging, **submit only one live run at a time**, and nothing on the board may be running when the judges arrive (see `checklist.md`).

## What's on the screen (so you can name it)

| Part | Where | What to say about it |
|---|---|---|
| Sidebar | left | "Every task is a conversation." Each row shows its status and best score. |
| Composer | bottom of a new task, or under the last turn | Text box, **Instant / Effort** toggle and **Model** picker (Opus 5.5, Sonnet 5.5, Haiku 5.5). Under it is a hint: "1 reward · first video in about 2 min" or "4 rewards side by side, live coach · about 4–5 min". |
| Step strip | top of a running turn | Set up task → Write rewards → Test code → Compile → Train → Record video → Judge. A plain-language line above it says what's happening now, plus a step timer and a whole-run timer. |
| Success check | under the status | "Pass = the robot reaches the ball and pushes it into the goal in at least 80% of episodes and falls in at most 10%." It's written before training and never changes. After the run there's a Metric / Got / Need / Result table with ✓ and ✕. |
| Best result | middle | 10 s video, critic score /10, "✓ Passed the success check", and the change since round 1. |
| Claude, judging the video | under the table | Starts with the phase breakdown (no_approach, no_push, wrong_way, short, overshoot, delivered). "Read Claude's full notes" opens the rest. |
| Coach, watching training | Effort runs only | Haiku's notes per candidate at each eval ("1.3M steps: walks forward at 0.5 m/s, but the ball is still 1.4 m short..."), ending with a "Coach: 12 looks, median 3.2 s, 0 stopped, $0.0071" line. **Instant runs have no coach.** |
| Refine this | on every scored candidate | Points the follow-up composer at that candidate. |
| Explore the run | collapsed rows | All candidates, the learning curve, the reward breakdown, **Reward function** (the Python Claude wrote), **What changed** (the editor's diff) and the activity log. |
| Export policy | under the score | Only if #50 is merged and deployed. Downloads a zip with the policy, the reward, the scene and `run_policy.py`. |
| Refusal panel | instead of a result | Only if #48 is merged and deployed. Shows a plain reason and the closest doable prompt. |

---

## 0. Opening state (set up at 7:45, re-check right before you're called)

Open these as browser tabs, left to right. Fill in the IDs in `golden-runs.md` after the morning runs.

1. **Tab 1, golden conversation** `#<G1 id>`: "push the ball into the goal", turn 1 Effort, plus a follow-up turn "get behind the ball first" that scored higher. Scroll so turn 1's video and the success-check table both fit on screen. Pause the video on frame 1.
2. **Tab 2, new task** `#new`, with nothing typed. Set the toggle to **Instant** now. The new-task default is Effort, and a 5-minute run would sink the pitch.
3. **Tab 3, pre-run candidate for a live follow-up** `#<G3 or G2 id>`, scrolled to a candidate's **Refine this** button.
4. **Tab 4, refusal** `#<G6 id>`, only if #48 is deployed.
5. **Tab 5, the hind-leg conversation** `#cca408b2` (best 3.0, check failed). This is the honest-failure example for Q&A.
6. **Local file:** `rewardsmith-demo.mp4` on the desktop, open in QuickTime and paused. This is the fallback (`demo-video.md`).

Browser zoom at 125% so the table reads from 2 m away. Turn off notifications. Plug in the laptop.

## 1. Golden conversation (pitch 0:00–0:15, then 1:15–1:50)

1. Tab 1. **Click play** on turn 1's best video. Let the hook run over it.
2. At 1:15, **scroll** to turn 1's success-check table. Point at **Got** for "Reach ball and Push ball into goal" and at "Episodes with a fall".
3. **Scroll** to the follow-up turn. Point at its header ("Follow-up · Instant · Sonnet 5.5 · refines Turn 1 · Round 1 · Candidate N") and at its table. The header shows that the follow-up continued from that exact candidate.
4. If there's time, point at **Coach, watching training** on turn 1 (Effort runs have it): "Haiku looks at every checkpoint while it trains, for about a cent per run."

## 2. Live Instant prompt (pitch 0:35, runs about 2.5 min)

1. Tab 2. **Click** the text box. **Type** the live prompt. The default is `walk to the red ball`. Use whichever passed rehearsal in `golden-runs.md` G5.
2. **Check that the toggle says Instant** and the hint says "1 reward · first video in about 2 min". Leave the model on its Instant default (Sonnet 5.5).
3. **Click Start training.** The turn opens and the step strip appears.
4. While it trains, say these lines in this order. The pitch already covers the first two, so use the rest during Q&A if the run is still going:
   - **Set up task** (a few seconds): "Claude just built the scene, a red ball somewhere in front of the robot at a random distance and angle each episode, and wrote the pass/fail test." **Point at the Success check line** when it appears.
   - **Write rewards / Test code** (under a minute): "It's writing the reward as Python and JAX. Every draft has to compile and survive a few simulator steps before it gets any GPU time." If the gate rejects a draft, the activity log says so. Mention it: "that's Claude fixing its own crash."
   - **Compile / Train** (the longest part; the PPO compile alone is about a minute): "PPO is compiling. That's the fixed cost: about a minute before the first training step. Then three million steps on top of the pretrained walker."
   - **Record video / Judge**: "Now it renders a 10-second rollout and measures 32 fresh randomized episodes. Then a second Claude looks at frames from the video and the numbers, and its score is capped by code if the test failed."
5. When the video appears, **press play**. Read the table out loud: "Reached the ball in X%, fell in Y%."

> Instant has **no live coach**, by design (one candidate, nothing to compare). For coach commentary, point at turn 1 of the golden conversation, which is an Effort run.

## 3. Follow-up on a pre-run candidate (Q&A, or right after the live run finishes)

Do this **only after the live Instant run has finished**, or it queues behind it.

1. Tab 3. **Click Refine this** on the candidate you picked in the morning (`golden-runs.md` G3: the hop or the trot).
2. The composer now says what it refines ("Refines Round 1 · Candidate 1 · Use the best instead"). **Type** the follow-up from `golden-runs.md` G3 (default: `lift your feet higher`). Make sure the toggle says **Instant**.
3. **Click Send** (the button reads **Keep going** when the box is empty and **Send** once you type). Say: "This doesn't start over. It continues the trained policy, the editor gets the critique plus my note, and the success check is copied from turn 1, so the goalposts stay put."
4. The turn header says "Follow-up · Instant · … · refines Turn 1 · Round 1 · Candidate N". Point at it.
5. With an **empty** box the button reads **Keep going** and sends exactly that: another round on the best.

## 4. Honest refusal (only if #48 is merged and deployed)

1. Tab 4, pre-run: "pick up the ball with your mouth". Show the refusal panel: "The Go1 has no mouth or gripper..." plus a suggestion ("walk to the ball and nudge it forward with your nose").
2. Say: "It refuses before spending any GPU time. On 32 judge-style prompts it classified 30 correctly and refused all 9 impossible ones."
3. If a judge wants to see it live: type it in a new task. It costs one Claude call and **no GPU**, so it's safe even while another run is queued. The refusal should land within seconds.

If #48 isn't deployed: don't type it. A pre-#48 board will try to train it. Say: "Honest refusals are in a PR that's tested but not deployed: 30 of 32 prompts classified right, 9 of 9 impossible ones refused."

---

## 5. Fallbacks

### A. The GPU is slow or the live run fails
- **Slow** (the step strip still says Compile or Train at 2:45): close the pitch on the golden conversation (pitch close, "If not" line). Don't refresh, and don't submit anything else. The result usually lands in the first minute of Q&A. Glance at tab 2 between answers.
- **Failed** (the step strip says "The run stopped with an error", or the check fails badly): say it straight: "That one didn't make it. Here's what the board does when it fails." Then **open Explore the run → Activity log** to show the error or the critic's diagnosis. Switch back to tab 1. Don't retry live during the pitch.
- **Stuck at Queued:** something else is on the worker. Say "the queue's busy", and tell the person on pod duty to check that nothing else was submitted. Show the golden tabs and the video.

### B. The board is unreachable (proxy down, venue Wi-Fi dead)
1. Try a phone hotspot (laptop on hotspot, reload tab 1) for ≤ 15 s.
2. If that fails, switch to the **local `rewardsmith-demo.mp4`** and say: "Venue Wi-Fi, so here's the recording from this morning. Same board, same runs." Narrate it with the pitch's words; the shots match the pitch order (`demo-video.md`).
3. Offer the URL and the QR code on the new-task page ("Scan to try it yourself") for judges to open later.

### C. A judge asks for a random prompt
Sort it in your head first (3 seconds), then:

| Kind of prompt | Examples | What to do |
|---|---|---|
| Locomotion or one object on flat ground | "walk in a circle", "go to the box", "push the ball left" | **Run it in Instant.** "About two and a half minutes. Let's watch." |
| Terrain or stairs/ramp | "walk on rough ground", "climb the stairs" | Rough terrain works on main. Stairs/ramps need #48. If #48 isn't deployed, say so and show the rough-terrain golden run. |
| Pose / acrobatics | "stand on your hind legs", "backflip", "sit" | Be honest: "That's our known weak spot." Show **tab 5** (`#cca408b2`): it reared up, the test failed it, and the score was capped at 3. "The point is that it doesn't lie about it." Run it if they insist. |
| Impossible for this body | "pick it up", "fly", "go to the red one" (colors shuffled) | With #48: run it; it refuses in seconds without GPU. Without #48: explain the limit (no gripper, no color vision) and offer the nearest doable prompt. |
| Two runs at once | "run mine and his" | One at a time. The second shows Queued and starts when the first finishes. |

Keep **Instant** for every judge prompt. Effort is about 5 minutes (4m 54s measured on the live board) and only fits if they're staying after the pitch.
