# Axiom — Screen Recording Walkthrough Script

> **How to use this:** Read this out loud while you record your screen. The lines in
> **[ON SCREEN: ...]** tell you what to be showing or clicking. The lines after them are
> what you actually say. Talk slowly, pause between sections. Total run time ~8–10 minutes.
> You don't have to read it word for word — it's a guide so you never go blank on camera.

---

## 0. Before you hit record (checklist)

- Start the app: open a terminal, go into the `frontend` folder, run `npm run dev`.
- Wait until it says the site is ready, then open **http://localhost:3000** in your browser.
- Have one CSV file ready on your desktop to upload (e.g. the credit-card fraud file).
- Close noisy tabs and notifications. Zoom the browser in a little so text is readable.

---

## 1. Opening — what is this thing? (≈45 sec)

**[ON SCREEN: the landing page / home screen of the app]**

"Hi everyone. I want to walk you through the project I've been building. It's called
**Axiom**.

The simplest way to describe it: **Axiom is an automatic data scientist.**

Normally, if a company has a spreadsheet of data and wants the computer to learn from it —
say, to predict which transactions are fraud — they'd need an expert who knows how to clean
the data, build a model, test it, and write up the results. That takes skill, and it takes
time.

Axiom does all of that **by itself**. You upload a spreadsheet, you tell it which column you
want to predict, and it does the entire job for you — and gives you a clean report at the
end. **No coding needed.**"

---

## 2. The big picture — how it's built (≈45 sec)

**[ON SCREEN: stay on the home page, or show a simple diagram if you have one]**

"Before I click around, here's the shape of it in one breath.

There are **two parts** working together:

- The **website** — the thing you're looking at. This is what a person actually uses. Clean,
  modern, runs in your browser.
- The **brain behind it** — a program running quietly in the background that does all the
  heavy data work. The website just sends it your file and shows you the results.

And one honest point I want to make early, because people always ask: **this does not use
ChatGPT or any AI chatbot.** The 'thinking' is done by classic, well-proven math and
statistics running on the machine itself. That means it's fast, it's private — your data
never leaves the computer — and it costs nothing per run."

---

## 3. The two modes (≈30 sec)

**[ON SCREEN: the screen where you pick Free vs Enterprise — the /welcome page]**

"When you come in, there are two ways to use it.

There's a **simple mode** — for someone who just wants to upload a file and get an answer.
Upload, watch it work, see the result. Three steps.

And there's an **advanced mode** — a full dashboard for power users, with extra tools to
customize things, look at history, and dig into details.

Let me start with the simple mode so you can see the whole thing end to end."

---

## 4. Uploading a file (≈45 sec)

**[ON SCREEN: go into the simple/Free flow, land on the upload screen]**

"Here's the upload screen. I'll drag in a spreadsheet. This one is a list of card
transactions, and somewhere in it is a column that says whether each transaction was fraud
or not.

**[ON SCREEN: drop the CSV file, wait for the preview to appear]**

As soon as I drop it in, notice it instantly shows me a **preview** of the data and a couple
of starter charts. So right away I can see it understood my file — the columns, the rows, all
of it.

**[ON SCREEN: point at / select the target column]**

Now the one thing I tell it: **which column do I want to predict?** Here, I pick the 'fraud'
column. That's it. That's the only decision I have to make. Everything else, Axiom figures
out on its own."

---

## 5. Watching the pipeline run — the heart of it (≈2 min)

**[ON SCREEN: start the run, the live progress view with the steps appears]**

"Now I hit run, and this is the part I'm most proud of. Watch the screen — you'll see it move
through a series of **steps**, live, one after another. Think of it like an assembly line.
Each step does one job and hands its work to the next. There are **eight** of them.

Let me narrate what each one is doing in plain English as they go by.

**[ON SCREEN: point at each step as it lights up]**

1. **First, it reads and understands the file.** It looks at every column and works out what
   it is — a number, a date, a category, true/false. And it figures out, on its own, what
   kind of question we're asking. Here it correctly sees this is a 'yes or no' question:
   fraud or not fraud.

2. **Second, it cleans the data.** Real data is messy — blank cells, duplicate rows, numbers
   stored as text like a dollar sign in front. It fixes all of that automatically, the way a
   careful analyst would.

3. **Third — and this is the clever one — it builds better clues.** It takes the raw columns
   and creates smarter versions. For example, from a date it pulls out the hour of day or the
   day of the week, because fraud often happens at odd hours. For card data it works out
   things like the distance between the customer and the store. It's basically giving the
   model better hints to work with.

4. **Fourth, it splits the data into three piles.** One pile to learn from, one to practice
   on, and one it locks away in a drawer and never looks at — until the very end. That last
   pile is how we get an *honest* score, because the model is being tested on data it has
   genuinely never seen. For fraud especially, it's careful to **train on the past and test
   on the future**, just like real life.

5. **Fifth, it trains a whole team of models at once** — seven or eight different approaches,
   all at the same time — and then holds a competition to see which one is best.

6. **Sixth, it audits the winner** — it double-checks for common mistakes, like a model that
   looks great on paper but would actually fail in the real world.

7. **Seventh, if it found anything to fix, it tries to tune the model and make it better** —
   but only keeps the change if it genuinely improves things.

8. **And eighth, it writes up everything** — the final honest score, the explanations, and a
   full report.

**[ON SCREEN: progress bar finishes / lands on the results screen]**

And... it's done. The whole thing — work that would take a person hours or days — just ran
in a couple of minutes."

---

## 6. The results (≈1 min 30 sec)

**[ON SCREEN: the results page with scores and charts]**

"Here's the payoff — the results page.

**[ON SCREEN: point at the headline score / metric]**

At the top is the headline: how well did the model do. And importantly, for fraud we don't
just show 'accuracy', because accuracy is misleading when fraud is rare. If only one in a
thousand transactions is fraud, a lazy model that says 'nothing is fraud' is 99.9% accurate
and completely useless. So we show the score that actually matters for catching rare events.

**[ON SCREEN: scroll to the operating-points table if visible]**

This part is built for a real fraud team. It says, in plain terms: *'if you investigate the
top 1% riskiest transactions, you'll catch this many percent of the actual fraud.'* That's
the number a manager actually cares about, because it maps to how many cases their team can
review in a day.

**[ON SCREEN: scroll to the feature-importance / SHAP chart]**

And this chart shows **why** the model made its decisions — which clues mattered most. So
it's not a mysterious black box. You can see what it paid attention to."

---

## 7. The report and exports (≈45 sec)

**[ON SCREEN: open the report view / show the download buttons]**

"Everything I just showed you also gets written into a **full report** you can keep or send
around. There's a nicely formatted document, you can **download it as a PDF**, and you can
even export it as a notebook for the technical folks who want to dig in or re-run it.

The report explains itself in plain English — what it did to the data, what it dropped, what
it added, and why it picked the model it picked. So someone who wasn't watching can read it
and understand exactly what happened."

---

## 8. The advanced mode — quick tour (≈1 min)

**[ON SCREEN: switch over to the Enterprise dashboard]**

"Quickly, let me show the advanced side for power users.

**[ON SCREEN: show the dashboard, then the workflow builder]**

This is the dashboard. The standout feature here is the **workflow builder** — those eight
steps I showed you earlier? Here you can see them as building blocks and rearrange them, or
run just the ones you want, instead of the whole assembly line. It's for people who want more
control.

**[ON SCREEN: click through history / runs and reports]**

There's also a **history** of every run you've done, so you can come back and compare, and a
place to revisit all your past reports. Same engine underneath — just more dials to turn."

---

## 9. Wrap up (≈30 sec)

**[ON SCREEN: back to the home page or the results screen]**

"So to sum it all up:

**Axiom takes a raw spreadsheet and turns it into a trained, tested prediction model and an
honest, readable report — automatically, in minutes, with no coding.**

It's especially good at fraud detection, where the tricky part isn't building *a* model, it's
building one you can actually *trust* — and a lot of the work I put in was specifically about
getting that honesty right.

That's the tour. Happy to go deeper on any part — thanks for watching."

---

## Quick cue card (for a second take, if you want it short)

If you need a 60-second version, just hit these beats:

1. "Axiom is an automatic data scientist — upload a spreadsheet, it builds and tests a
   prediction model for you, no coding."
2. "It runs an 8-step assembly line: read → clean → build clues → split → train many models →
   audit → tune → report."
3. "It doesn't use any AI chatbot — it's classic math, runs locally, private and free per run."
4. "It's built for fraud: it shows the score that matters for rare events, and tells a team
   exactly how much fraud they'd catch."
5. "You get a full report — readable, downloadable, explains itself."
