# Morning checklist, Sun Oct 25 (all times PT)

**Hard constraints:** at the Palace of Fine Arts and checked in by **8:00**. Devpost closes at **9:00**, hard; we submit by **8:30**. Judging: 3-minute pitch + 2 minutes of Q&A.

Board: https://o95f0trj2xjkjb-8000.proxy.runpod.net · Pod SSH: `root@81.27.69.179 -p 28002` (runpodctl key)

## Tonight, before sleeping (by 2:00)

- [ ] **Decide what ships.** Open PRs: #48 (scenes, gate, refusals), #49 (code checks, stacked on #48), #50 (export). Anything not merged by the 4:30 cutoff stays out of the demo. The docs already say what to cut for each.
- [ ] **Confirm the roles below** with Corey, Jiacheng and Joshua.
- [ ] **Read `pitch.md` aloud once with a stopwatch.**
- [ ] Set alarms: 4:15 (whoever deploys), 5:00 (whoever runs the golden runs).

## Morning

| Time | Who | What | Done when |
|---|---|---|---|
| **4:30** | Yihan | **Final merge cutoff.** Nothing merges after this. Tell the other Claude sessions to stop all benchmarks on the pod. | `nvidia-smi` on the pod shows only the live worker, or nothing. |
| **4:45** | Yihan | **Deploy main to the pod** and restart the server and worker with `/workspace/restart_live.sh`. Run it yourself: the permission classifier blocked a Claude session from killing the old processes last time. | `GET /api/health` is 200 with `"fake_sim":false,"fake_llm":false`. The board loads. No run shows Queued. |
| 4:50 | Yihan | **Smoke test:** one Instant `walk backwards`. | Video + score in under 3 min. |
| **5:00–6:45** | Yihan or the main session | **Pre-run the golden conversations,** in order (`golden-runs.md`). Screen-record G1a while it runs (`demo-video.md` shot 3). | IDs written into `golden-runs.md`; the live prompt chosen (G5). |
| 6:00–7:15 | Joshua | **Cut the 2-minute video** from the finished conversations (`demo-video.md`). Upload it to YouTube unlisted. | Link pasted into the Devpost draft; `rewardsmith-demo.mp4` on the pitch laptop's desktop. |
| 6:00–7:00 | Corey | **Fill in the Devpost draft** from `devpost.md`: add all 4 teammates, the gallery image (a frame of the G1 video, 3:2), and the tracks (Anthropic, Most Technically Complex, Most Creative). | Draft saved, everything but the video link and repo link. |
| **6:45** | Yihan | **GPU quiet.** No new runs except judge prompts. | |
| **7:00** | Yihan | **`touch /workspace/live.hold` on the pod.** It pauses every benchmark until removed. The pause daemon lives on the pod only and doesn't survive a container restart, so check it's still running after the 4:45 restart (`ls -l /workspace/live.hold`, then ask the main session). Leave it in place until judging ends. | The file exists. |
| **7:00** | **Yihan (only he can)** | **Make `yih0nk/rewardsmith` public.** First run a quick secret scan: `git log -p \| grep -nE 'sk-ant-\|ANTHROPIC_API_KEY=.+'` should only hit `.env.example`. On GitHub: Settings → General → Danger Zone → Change visibility → Public. | The repo opens in a private browser window. |
| 7:15 | everyone | Leave for, or already be at, the Palace of Fine Arts. | |
| **8:00** | everyone | **Checked in at the venue.** | |
| 8:00–8:15 | Corey | Paste the repo URL and the video link into Devpost. Proofread it once on the phone. | |
| **by 8:30** | **Yihan + Corey** | **Submit on Devpost.** All 4 teammates are listed and the tracks are ticked. If anything is missing at 8:25, submit anyway and edit later; Devpost allows edits until the 9:00 deadline. | Devpost shows "Submitted". |
| 8:30– | Jiacheng | **Pitch laptop ready:** charged to 100% plus the charger; tabs open as in `golden-runs.md` "Tabs at 7:45"; zoom 125%; Do Not Disturb on; the `#new` toggle on Instant; `rewardsmith-demo.mp4` on the desktop; phone hotspot tested. | |
| before the pitch | Jiacheng | Reload every tab. Check that nothing on the board is Queued or running. Pause tab 1's video on frame 1. | |
| after judging | Yihan | `rm /workspace/live.hold` | |

## Who says what (suggested; confirm tonight)

| Person | Pitch | Q&A | During the run |
|---|---|---|---|
| **Yihan** | Speaks the whole 3 minutes (`pitch.md`). | Technical questions: Eureka/L2R/RT-2, the walker, the bugs, sim-to-real, limits. | — |
| **Joshua** | **Drives the laptop:** every bracketed click in `pitch.md`, typing the live prompt. | Board and UX questions; runs the follow-up and refusal demos (`demo-script.md` §3–4). | Points the cursor at what Yihan names. |
| **Corey** | — | "How do you know it worked": the success checks and per-episode outcomes (he built the scene-aware checks, #37). | Holds the stopwatch and signals 2:30 and 2:50. |
| **Jiacheng** | — | Cost, latency and infrastructure (Runpod, the 4090, the queue). | **Pod duty:** watches tab 2, says "still training" or "done". Owns fallback B (hotspot, then the local video). |

Swap roles if someone is better at a part. The one rule: **the speaker never touches the laptop.**

## If something breaks in the morning

- **The deploy fails or the board is down at 5:00.** Roll back to the backup in `/workspace/rs_live_backup_*` and the previous code. If that also fails by 6:00, demo from the video plus the existing live conversations (`#9d26b109` turn 1, `#5b84e1b9`, `#7611b1fe`) and skip the live prompt.
- **Golden runs keep missing.** Use `#9d26b109` **turn 1 only** (8.0, 96.9%, 0 falls) as tab 1. Its turns 2–3 are on the same page and scored 3.0, so don't scroll past turn 1.
- **Running late.** The order that matters: Devpost submitted > checked in by 8:00 > video > golden follow-up > everything else.
