# 2-minute demo video

**Purpose:** two jobs. (1) Devpost asks for "a clear, meaningful video demo" for sponsors judging asynchronously. (2) It's the fallback if the board is unreachable during the pitch (`demo-script.md` §5B).

**Rule: nothing waits on the GPU.** Every shot uses a conversation that has already finished, except shot 3. Shot 3 is screen footage of the G1a Effort run recorded **while it runs at 5:00** (`golden-runs.md`), then sped up.

## How to shoot

- **Capture:** QuickTime → New Screen Recording, browser window only, 1920×1080, zoom 125%. Hide bookmarks and extensions. Turn on Do Not Disturb.
- **Narration:** record it separately, in a quiet corner with the phone's voice memo, reading the script below. Lay it over the footage. Two clean takes beat one live take.
- **Edit:** iMovie or CapCut. Plain hard cuts, no music under the voice, text labels in IBM Plex Mono if you add any. Export 1080p.
- **Upload:** YouTube **unlisted**. Paste the link into Devpost. Also keep `rewardsmith-demo.mp4` on the pitch laptop's desktop.
- **Length:** 1:55–2:00. Narration runs about 290 words.

## Shots

| # | Time | Screen (what's recorded) | Narration |
|---|---|---|---|
| 1 | 0:00–0:10 | **G1 conversation**, best video playing full width: the Go1 walks to the orange ball and pushes it into the green zone. | "This robot dog learned to push a ball into a goal from one English sentence. Nobody wrote its reward function. It took about five minutes." |
| 2 | 0:10–0:22 | **`#new`**: slowly type `push the ball into the goal`. Show the Instant/Effort toggle and the model picker. **Do not click Start training.** Cut away before the click. | "Normally, teaching a robot something new means hand-tuning a reward for days, or collecting thousands of demonstrations. In Rewardsmith you type the behavior and pick how hard to search: Instant for one try in about two minutes, Effort for four side by side." |
| 3 | 0:22–0:50 | **G1a while running** (recorded at 5:00), sped up about 8×, with a small "8×" label. Pan across the step strip (Set up task → Write rewards → Test code → Compile → Train → Record video → Judge), the success-check line as it appears, and the **Coach, watching training** feed filling in. | "Claude builds the scene and writes a pass/fail test before any training, so it can't move the goalposts later. Then it writes four reward functions as code. Each must compile and run in the simulator; if one crashes, Claude reads the error and fixes it. PPO trains them on a GPU, on top of a walker we pretrained, and a fast Claude coach comments on every checkpoint." |
| 4 | 0:50–1:10 | **G1, finished turn 1**: the score and "✓ Passed the success check", then a scroll to the Metric / Got / Need / Result table, hovering on "Reach ball and Push ball into goal". Open **Read Claude's full notes** for 2 s. | "Every policy runs 32 fresh, randomized episodes against that test. Here the ball ends in the goal in [N] percent of them, with no falls. A second Claude watches the video and scores it, and if the test fails, code caps the score, so a nice-looking video can't hide a miss." |
| 5 | 1:10–1:25 | **G1, follow-up turn**: zoom on the header "Follow-up · Instant · refines Turn 1 · Round 1 · Candidate N", then its video and table. *(If G1b wasn't kept, use the hop conversation G3 and the narration in brackets.)* | "And it's a conversation. We said 'get behind the ball first'. The follow-up continues from the trained policy, and the ball reached the goal more often." [*Alt:* "And it's a conversation: refine any candidate, or just say keep going, and it continues from that trained policy."] |
| 6 | 1:25–1:40 | **Quick cuts, 4–5 s each:** G3 hop video, G4 rough-terrain video, and the G6 refusal panel if #48 is deployed (otherwise the hind-leg `#cca408b2` failed check). | "Same loop for hopping, rough terrain, anything this body can do. And it's honest: when it can't do something, it fails the test or tells you why, instead of passing a crouch." |
| 7 | 1:40–1:52 | **Text card** on black, mono font: `push: 15.6% → 72–100%` · `falls: 37.5% → 0–3%` · `critic: 2 → 8` · `Instant ≈ 2.5 min · Effort ≈ 5 min` | "On our hardest prompt, the push went from working in sixteen percent of episodes to seventy-two to a hundred, and falls dropped from thirty-eight percent to three or less." |
| 8 | 1:52–2:00 | **Text card:** `Rewardsmith` · `same loop, any body` · repo URL | "Same loop, any body. Rewardsmith: describe a behavior, get a robot that does it." |

## Backup if there's no time for an edit

Record one continuous take of the golden tabs (shots 1, 4, 5, 6) while reading the narration live, 2:00 max, with no cuts. It's worse than an edit but much better than no video. Do this by **7:30 at the latest**.
