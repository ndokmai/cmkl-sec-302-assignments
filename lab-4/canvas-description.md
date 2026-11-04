## **Lab 4: Evasion Attacks with FGSM**

---

## **Goal**

Find **two different images** that demonstrate different outcomes of an FGSM attack
and a median-filter defense:

- **Image 1:** the attack succeeds, but the defense fails.
- **Image 2:** the attack succeeds, and the defense succeeds.

Experiment with different epsilon values and prepare to discuss in class why the
defense helps in one case but fails in the other.

---

## **Instructions**

No coding is required. Use the controls to generate attacks and explore the results.

1. Visit the [Lab 4 GitHub folder](https://github.com/ndokmai/cmkl-sec-302-assignments/tree/main/lab-4).
2. Clone the repository, or pull the latest changes if you already have it.
3. Open the `lab-4` directory and follow its README to launch the lab in **Marimo app mode**.
4. Find **two different images of your own**. No example image is provided.
   Upload each as a JPEG/PNG. Choose images whose clean predictions correctly
   identify the main object.
5. **Use the smallest noise that makes the attack succeed.** Start with a small
   epsilon and increase it gradually until the prediction changes. Then reduce
   epsilon to find the **smallest successful value you can select with the slider**.
   Repeat this search for each image. Report it as the smallest value you tested,
   rather than claiming it is the exact minimum.
6. **Preserve as much image detail as possible while stopping the attack.**
   At your selected epsilon, try enabled median filters from smallest to largest:
   **3 × 3, 5 × 5, then 7 × 7**. Choose the smallest filter that restores the correct
   clean class and keeps the filtered clean prediction correct. Compare visible
   detail before and after filtering and explain your choice. For the defense-failure
   case, report that none of the tested filters meets these conditions if that is
   what you observe; do not claim a successful defense. Record the
   clean, adversarial, filtered adversarial, and filtered clean predictions,
   together with epsilon and filter size. Inspect the noise and visible distortion.
7. Find the two required cases using **nonzero epsilon** and an **enabled filter**:
   - **Image 1 · Attack succeeds, defense fails:** the adversarial prediction changes
     from the correct clean class, and filtering does **not** restore that class.
   - **Image 2 · Attack succeeds, defense succeeds:** the adversarial prediction
     changes, and filtering restores the correct clean class.
8. Check the **filtered clean control** for both images. Aim for settings that
   preserve the clean prediction; if filtering alone changes it, report that
   explicitly because the defense is also damaging the clean input.
9. Download both comparison images for each case.
   If possible, also compare the two images at the same epsilon and filter size
   to help separate image differences from parameter differences.

A confidence decrease alone does **not** count as an attack success. The defense
succeeds only when it restores the original correct class; changing the wrong
prediction to another wrong class does not count.

---

## **Share your examples on Padlet**

Post the **FGSM and median-filter comparison images for both cases** on the
[class FGSM Padlet wall](https://padlet.com/natnateedokmai/submission-request/PW6VbOQZVAaKXY19).
Label them **Image 1 · Defense fails** and **Image 2 · Defense succeeds**.
Include each image's **epsilon**, **filter size**, and
**clean → adversarial → filtered adversarial** predictions.

---

## **Submission**

Submit the following on Canvas for **both of your images**:

- **Two FGSM comparison images**, one per case, showing clean + noise = adversarial
  and readable predictions.
- **Two median-filter comparison images**, one per case, showing the defense result
  and the filtered clean control, with readable predictions.
- A short **results table** covering the different epsilon values you tried for
  each image. Include image/case, epsilon, filter size, clean prediction,
  adversarial prediction, filtered adversarial prediction, and filtered clean
  prediction. Include unsuccessful trials as well as the chosen settings.

---

For each image, identify your **smallest tested successful epsilon** in the results
table. If the two required defense outcomes need different settings, also label
the epsilon and filter size used for each chosen case.

---

## **Prepare for the in-class discussion**

Bring both examples and your results table. Be ready to explain:

- Which visual features distinguish each object or class: overall shape, distinctive
  parts, texture, or fine details?
- Could median filtering remove the attack noise while preserving those features,
  or could it also remove information needed for classification?
- Does distinguishing closely related classes, such as dog breeds, seem different
  from identifying an object with distinctive features? Use your observations to
  support your explanation rather than assuming this always holds.
- Which filter preserved the most visible detail while stopping the attack?
  Could a smaller filter also succeed?
- How did epsilon and filter size change the outcome? Did the filtered clean
  control reveal any loss of useful information?

Treat these explanations as **hypotheses supported by your experiments**. Two images
cannot establish a general rule about which object categories the defense protects.
