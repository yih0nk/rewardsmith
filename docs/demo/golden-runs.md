# Golden runs: pre-run on the live board, 5:00–7:00 AM PT

**Who runs these:** Yihan, or the main session, **after the final merge and deploy** (see `checklist.md`, cutoff 4:45). Nobody runs them before that, so the live board isn't disturbed overnight.

**How:** type each prompt into the board's composer (`#new`, or a conversation's Follow up box), exactly as written. The worker is serial on one RTX 4090, so submit the next one only when the previous one says Finished. Queued runs work too, but if one goes bad you can't change course.

**Timing used below:** Effort is one round of 4 rewards, measured at **4m 54s** on the live board (request 96acf8a0). Instant is one reward, measured at **2m 23s–2m 34s**. Leave about 30 s between runs.

**A run is "golden" when all of these hold:**
- ✓ Passed the success check, and every row of the table passes
- the critic score is 7 or higher
- "Episodes with a fall" is 10% or less
- in the 10 s video, the behavior is obvious within the first 5 seconds to someone who has never seen the board

If a run misses, retry it **once**. If the retry also misses, drop it and move on. Never show a run that looks worse than the one before it.

Write each run's id (the `#xxxxxxxx` in the URL) into the **ID** column as you go. `demo-script.md` and `demo-video.md` refer to these rows.

| # | Start | Mode | Prompt (type exactly) | ID | Keep in tab |
|---|---|---|---|---|---|
| G1a | 5:00 | **Effort** | `push the ball into the goal` | | Tab 1 |
| G1b | 5:06 | Instant, follow-up on G1a's best | `get behind the ball first` | | Tab 1 (same conversation) |
| G2 | 5:12 | **Effort** | `walk to the blue box` | | Q&A spare |
| G3a | 5:18 | **Effort** | `hop with all four feet leaving the ground at the same time` | | Tab 3 |
| G3b | 5:24 | Instant, follow-up (rehearsal) | `lift your feet higher` | | (rehearsal only) |
| G4 | 5:28 | **Effort** | `walk forward on rough terrain` | | Q&A spare |
| G5 | 5:34 | **Instant** ×3 | `walk to the red ball` (×2), `walk backwards` (×1) | | none (rehearsal) |
| G6 | 5:43 | Instant | `pick up the ball with your mouth` (**only if #48 is deployed**) | | Tab 4 |
| G7 | 5:44 | **Effort** | `walk up the stairs` (**only if #48 is deployed**) | | Q&A spare |
| — | 5:50–6:45 | | Retries for anything that missed, G1 first | | |
| — | 6:45 | | **GPU quiet.** Nothing new is submitted after this. | | |

---

## G1: push the ball into the goal (the pitch's spine)

- **G1a, Effort.** The scene should be an orange ball 1–1.5 m out and a green goal zone about 3 m out. The check should read like "push:ball:goal ≥ 80%, scene_fell ≤ 10%". If the check shows only "don't fall" (fell / upright / height), **stop and re-run**: that check proves nothing, and it's exactly the failure we fixed in #39/#47.
- **Good result:** "Reach ball and Push ball into goal" at 80% or more, falls 0–10%, critic 7–8. The reference: #39/#47 measured push succeeding in 72–100% of episodes with 0–3% falls, and 9d26b109 on the live board got 96.9% and 0% falls (critic 8.0). The phase line under "Claude, judging the video" should say mostly "delivered".
- **G1b, follow-up.** Click **Follow up on this run's best** (or **Refine this** on the best candidate), type `get behind the ball first`, set **Instant**, and click **Send**.
  - **Keep it only if it improves**: a higher "Reach ball and Push ball into goal" %, or the same % with a higher score or fewer "wrong_way"/"short" phases. On the chat-runs test server this note took ball-reached-goal from 78% to 100%.
  - **Known risk:** on the live board, two follow-ups on 9d26b109 ("stop the ball inside the goal…") went from 8.0 to 3.0 (overshoot in 72–88% of episodes). A separate session is debugging follow-ups right now. If G1b doesn't improve: retry once with `line up behind the ball, then push it straight into the goal` (Instant). If that also fails, **show G1a alone** and cut the follow-up sentence from the pitch (`pitch.md` notes).
  - G1a is near 100% already, so it has little room to improve. If G1a lands at 100%, run G1b anyway, but judge it on the score and the video, and expect a tie at best. A tie is still useful ("it kept it at 100% and stopped wandering"); a drop isn't.

## G2: walk to the blue box

- **Use this exact wording, without "and stop".** #39 found that the `stand_still` part of an "...and stop" check can't pass under the sampled-action eval: a perfectly standing Go1 shows gyro spikes above the limit. Every "stop beside it" policy arrives and nearly stops (0.05–0.08 m/s) and still scores 0–3% on that clause. "Walk to the blue box" gets a reach check (`target_reached` or `reach:blue_box`) instead.
- **Good result:** reached 80% or more, falls ≤ 10%, and the video walks straight to the box. (Reference: walk to the red ball reached 32/32 episodes in text-to-task testing.)

## G3: hop with all four feet

- A re-run of 5b84e1b9 (8.0, check passed on older code). It runs fresh so the candidates have checkpoints on disk: a follow-up can only continue a trained policy if its `params_*.pkl` exists, and older candidates don't have one.
- **Good result:** the check passes on front_feet_off / rear_feet_off / feet_off_mean, plus upright and no fall. The video visibly bounces.
- **G3b is a rehearsal** of the demo's live follow-up (`demo-script.md` §3). If `lift your feet higher` doesn't improve or makes it worse, pick a different note now and write it here: ________. If no note helps, skip the live follow-up in the demo.
- If G3a misses twice, use the trot instead: `trot forward as fast as you can without falling` (Effort). The live board's 7611b1fe got 8.0 and passed on older code.

## G4: walk forward on rough terrain

- The rough heightfield is on main. #39 fixed the gate crash for rough terrain without objects: forward speed 0.45–0.92 m/s, critic 7–8, 0 of 4 fell.
- **Good result:** passes forward_velocity ≥ 0.3 and scene_fell ≤ 10%. The live board's older 42861399 got 7.0 but failed its check. Don't show that one.

## G5: rehearse the live prompt

- Run `walk to the red ball` in **Instant** twice and `walk backwards` in **Instant** once. Time each one from **Start training** to the video appearing.
- **Pick the live prompt:** use `walk to the red ball` if both runs passed the check and each took under 2m 45s. Otherwise use `walk backwards`, which is the most reliable behavior in every regression run. Write the choice in `pitch.md` (`LIVE_PROMPT`) and tell the speaker.
- These runs stay in the sidebar. That's fine.

## G6: honest refusal (only if #48 is merged and deployed)

- Expect a refusal panel within seconds, with no Train step: "The Go1 has no mouth or gripper…" plus a nearest doable prompt. If it starts training instead, #48 isn't live. Click Cancel and drop the refusal from the demo.

## G7: stairs (only if #48 is merged and deployed)

- #48 adds `stairs`, `ramp`, `wall`, `gap` and `field` structures, but **its GPU results weren't in the PR when we wrote this**. Treat this as a probe: keep it only if it passes cleanly. If it fails, don't mention stairs as working.

---

## Tabs at 7:45

1. G1 conversation (G1a + G1b if kept), scrolled to turn 1's video
2. `#new`, toggle on Instant, nothing typed
3. G3 conversation, scrolled to the best candidate's **Refine this**
4. G6 refusal (if any)
5. `#cca408b2`: hind legs, best 3.0, check failed (the honest-failure example)
6. G2 and G4 in one extra tab for Q&A, if wanted

## What this costs

About 7 Effort runs plus 6 Instant runs is roughly 50 min of GPU on the 4090. The coach is about $0.01 per Effort run. Opus/Sonnet writing and judging is the larger Claude cost; we haven't metered it per run.
