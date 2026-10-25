# The 3-minute pitch

About 430 spoken words, which is 3:00 at a brisk-but-clear 145 words a minute. **Bold brackets** are what the driver clicks. The speaker never touches the laptop. Rehearse with a stopwatch twice, out loud.

**Before the judges arrive:** the board is open on the golden conversation (`golden-runs.md`, tab 1). Its best video is paused on frame 1, with the success-check table visible under it. The new-task page is ready in tab 2, with nothing typed.

The live prompt (`LIVE_PROMPT` below) is whatever passed its two morning rehearsals in `golden-runs.md` (G5). The default is **"walk to the red ball"**.

---

### 0:00 – 0:15 · Hook

**[Driver: press play on the golden video.]**

> This robot dog has never seen a ball before tonight. Nobody on our team wrote a line of code for this. We typed one sentence: "push the ball into the goal." Five minutes later, it pushes the ball into the goal in **[N]** out of 32 randomized test runs.

*(Before you go up, fill in [N] from the golden run's success-check row, which shows a percentage of 32 episodes: 97% is 31. If it's 100%, say "in every one of 32 randomized test runs.")*

### 0:15 – 0:35 · The problem

> Today, teaching a robot something new means one of two things. Hand-write a reward function, which means an RL engineer tuning numbers for weeks until the robot stops cheating. Or collect thousands of demonstrations for a big generalist model. Either way, "try a new behavior" is a project, not a sentence.

### 0:35 – 0:50 · Start the live run

**[Driver: switch to tab 2. Type `LIVE_PROMPT`. Confirm the toggle says Instant. Click Start training.]**

> So let's try one now. Instant mode: one reward, about two and a half minutes. Claude has already started.

**[Driver: as soon as the turn appears, point the cursor at the step strip.]**

### 0:50 – 1:15 · How it works, in one breath

**[Driver: hover over "Set up task", then over "Success check" when it fills in.]**

> First, Claude builds the scene and writes a pass/fail test, before any training, so it can't move the goalposts later. Then it writes reward functions as JAX code, and if one crashes in the simulator, Claude reads the error and fixes it. PPO trains on a GPU, on top of a walker we pretrained, so the robot learns the task, not how to walk. Then a second Claude watches the video and scores it.

### 1:15 – 1:50 · The golden conversation

**[Driver: switch to tab 1. Scroll to the first turn's success-check table, then to the follow-up turn.]**

> Here's this morning's run. The test it wrote: ball in the goal in at least 80% of 32 randomized episodes, at most 10% falls. Then you follow up, like you would with a person. We typed "get behind the ball first."

**[Driver: point at the follow-up turn's table and score.]**

> The follow-up continues from the trained policy instead of starting over. On our test server, that note took the ball reaching the goal from 78% to 100%.

### 1:50 – 2:20 · Hard numbers

> The numbers. On the same prompt, our first version scored 2 out of 10. The push worked in 16% of episodes and the robot fell in 38%. Now it pushes the ball in 72 to 100% of episodes, with 0 to 3% falls. The biggest bug: the policy could see an episode clock it never trained past 6.4 seconds. Removing it took flips from 100% of episodes to 3%.

### 2:20 – 2:45 · Why it's different

> Eureka showed that LLMs can write rewards, but a human still writes each task's environment and success measure, and its search runs many full trainings. Language-to-Rewards fills in a fixed reward template for a controller. Generalist models like RT-2 and π0 need huge demonstration datasets. Rewardsmith writes the scene, the test and the reward, checks its own work against that test, and hands you a trained policy in minutes. It also tells you no: ask it to pick up a ball with its mouth and it explains that a Go1 has no mouth.

### 2:45 – 3:00 · Close

**[Driver: switch to tab 2. If the live video is in, press play. If not, leave the step strip on screen.]**

> *(If the live run is done:)* And there's the one you just watched us type.
> *(If not:)* That's still training, and we'll show it during questions.
>
> Same loop, any body: today a quadruped, next an arm or a humanoid. Rewardsmith. Describe a behavior, get a robot that does it.

---

## Notes for the speaker

- **The live run won't finish inside 3:00** if you start it at 0:40, because Instant measured 2m 23s to 2m 34s end to end. That's fine. The close works either way, and the result is ready by the first Q&A answer. Don't stall waiting for it.
- **Say numbers exactly as written.** Every one is sourced in `qa.md` § Numbers. If you blank, say "most" and keep going rather than guessing.
- **The refusal line in 2:20–2:45 depends on #48.** If #48 isn't merged and deployed, cut it ("...hands you a trained policy in minutes.") and save the refusal for Q&A as roadmap.
- **The 78% → 100% follow-up number was measured on the chat-runs test server, not the live board.** On the live board, two "stop the ball inside the goal" follow-ups on 9d26b109 made it worse: it overshot the goal in 72–88% of episodes. If this morning's golden follow-up didn't improve, drop the "78% to 100%" sentence and say: "The follow-up continues from the trained policy instead of starting over." Never show a follow-up that got worse.
- **If a judge interrupts,** answer and skip ahead to "Why it's different". The close matters more than the numbers.
