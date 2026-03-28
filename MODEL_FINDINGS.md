# What we actually learned from the models

This is a plain-language note for someone who is not a data scientist. The numbers come from our last notebook runs (same cleaned file for everyone: `clean_data_basic.csv`). If you change the data or the code, run the notebooks again and update the figures here.

---

## First: what problem are we solving?

Imagine a streaming company (or any subscription business). Some people always lean toward *“give me the cheap / simple option now”* and others lean toward *“I’ll pay more or wait if the long-run deal is better.”*  

In the dataset, that’s coded as **SS** = smaller reward sooner (kind of “impatient”) and **LL** = larger reward later (“patient”). The models try to guess which choice a person will make on a given offer, using things like amounts, delays, and a few behavior-style signals.

We trained on some people and tested on **other** people (we never mixed the same person between train and test). That’s important so we’re not cheating.

---

## How much data was it?

Roughly **one million rows** of choices after loading. For the tree models we dropped rows with missing pieces in the feature set and landed around **924k** rows.  

On the test side we had on the order of **190k individual choices** from about **2,100 people** the model had never seen during training — so when we say “accuracy,” it’s about those held-out decisions.

---

## Model 1 — Logistic regression (the simple, explainable one)

Think of this as drawing straight-line rules: “if this goes up, the chance of choosing LL goes up or down.”

**What it did on our run:** accuracy around **56%** on the test set — only a little better than always guessing the most common label (~55%). The **ROC-AUC** was about **0.68**, which means it sorts “later vs sooner” choices *better than a coin flip*, but it’s not amazing.

**What you can tell your aunt (or a product manager):**  
When the **later option is much bigger** than the sooner one (high *reward ratio*), people tilt toward waiting. When the **wait is longer**, they tilt back toward taking money now. That matches common sense — and thanks to this kind of model in a real company you could say: *“We have a transparent baseline: price and delay really do move the needle in the expected direction, even before we go fancy.”*

---

## Model 2 — Random Forest

This one allows curved, steppy patterns instead of one straight line.

**On our run:** about **69%** of test choices guessed right, **ROC-AUC ~0.76**. The simple logistic on the *same* test was more like **66%** accuracy and **0.72** AUC — so the forest genuinely helped.

**What showed up as “important”:**  
The model leaned heavily on **patience_rate** (how “patient” that person looked on *other* trials in the training period) and **log_discount_k** (a number related to how they trade off time vs money). After that came things like **reward ratio** and **how much extra delay** there is.

**In a real company you could deduce things like:**  
*“Past behavior is a strong hint for the next choice — so for CRM we might care about early signals of ‘always picks the immediate deal’ vs ‘comfortable waiting.’”*  
Also: *“The shape of the offer still matters after we account for the person — it’s not only personality.”*

The confusion matrices we saw: logistic regression was **too eager to say LL**; the forest caught **more true “take it now”** people correctly. For marketing, that’s the difference between annoying someone with the wrong upsell and spotting who actually responds to *“save today.”*

---

## Model 3 — XGBoost

Very similar story to Random Forest on our numbers: **~68%** accuracy, **ROC-AUC ~0.76**, a bit better than the logistic baseline. Same big picture on feature importance — **patience**-style signals and **discounting** on top, then the offer structure.

**So in a meeting you might say:**  
*“Two different tree methods agree on what matters. That makes us more confident it’s not a weird accident of one algorithm.”*

---

## Jargon cheat sheet (beginner)

- **Accuracy** — Out of all test choices, what fraction did we label right? Easy to understand, but it hides imbalance between SS and LL.
- **ROC-AUC** — Roughly: how well the model **ranks** who is more likely to pick LL vs SS. **0.5** = useless, **1.0** = perfect. **0.76** = decent sorting, not perfect.
- **Precision / recall (for LL)** — *Precision:* when we say “this person will pick LL,” how often were we right? *Recall:* of everyone who truly picked LL, how many did we catch?

---

## One paragraph you can steal for the presentation

“We used over a million experimental choices and held out whole participants for testing. A simple logistic model gave an honest story — bigger later rewards pull people toward waiting, longer waits pull them back — but predictions stayed weak. Random Forest and XGBoost did noticeably better on the same test people and agreed that **who you are** (patience habits, how you discount the future) plus **how the offer is built** drives choices. In a real subscription business we’d use that to think about **segments**, **which message fits whom**, and **which tier or promo to try first** — always paired with A/B tests, not blind automation.”

---

*Small print: if your XGBoost run used the sklearn fallback instead of real XGBoost, or you changed random seed, the decimals might wiggle slightly — trust whatever your notebook printed last.*
