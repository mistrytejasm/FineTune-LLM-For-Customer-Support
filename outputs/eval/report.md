# LLM Fine-Tuning Evaluation Report

> **Generated:** 2026-04-09 07:22  
> **Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
> **Adapter:** `outputs/adapter`  
> **Eval samples:** 50  

---

## Executive Summary

✅ **Fine-tuning was SUCCESSFUL.** The model clearly improved on domain-specific customer support responses.

| | Base Model | Fine-Tuned | Delta |
|---|---|---|---|
| **ROUGE-1** | 0.3485 | 0.5034 | 🟢 +0.1549 |
| **ROUGE-2** | 0.1035 | 0.2358 | 🟢 +0.1323 |
| **ROUGE-L** | 0.2152 | 0.3417 | 🟢 +0.1265 |
| **BLEU**    | 0.0536   | 0.1984   | 🟢 +0.1448   |
| **BERTScore F1** | N/A | N/A | N/A |
| **Exact Match** | 0.0000 | 0.0000 | ⚪ 0.0000 |
| **Approx Perplexity** ↓ | 4.80 | 2.97 | 🔴 -1.8300 |
| **Avg Response Length (words)** | 58.2 | 77.2 | — |
| **Avg Generation Time (sec)** | 7.506 | 11.971 | — |

---

## Charts & Visualisations

### Metric Comparison Bar Chart
*Side-by-side comparison of all metrics between base and fine-tuned model.*

![Metric Comparison Bar Chart](plots/metrics_comparison_bar.png)

### Training Loss Curve
*How the model's loss decreased over training steps.*

![Training Loss Curve](plots/training_loss_curve.png)

### ROUGE Score Radar Chart
*All four ROUGE sub-scores visualised as a spider/radar chart.*

![ROUGE Score Radar Chart](plots/rouge_breakdown.png)

### Response Length Distribution
*How verbose each model is compared to the ground-truth references.*

![Response Length Distribution](plots/response_length_dist.png)

### Per-Intent ROUGE-1
*Where the model improved most (by customer service intent).*

![Per-Intent ROUGE-1](plots/per_intent_rouge.png)

### Improvement Heatmap
*Green = improved, Red = regressed, compared to base.*

![Improvement Heatmap](plots/improvement_heatmap.png)

---

## Training Statistics

| Metric | Value |
|---|---|
| Initial train loss  | 2.3306 |
| Final train loss    | 0.6371 |
| Best eval loss      | 0.6452 |
| Training steps      | 313 |

---

## Side-by-Side Response Examples

### Example 1
**Intent:** `registration_problems` | **Category:** `ACCOUNT`

**Prompt:** i cant register can ya help me to report a signup error

**Reference (ground truth):**
> We apologize for the inconvenience you're facing with the registration process. Your decision to seek assistance demonstrates your determination to resolve the issue and we commend you for that. We understand the frustration caused by not being able to register, and we're here to assist you in repor...

**Base Model:**
> I'm sorry to hear that you're having trouble registering. Could you please provide more details about the issue? Specifically, what kind of error message did you receive when trying to sign up? This information will help us better understand your problem and assist you in resolving it. Thank you for...

**Fine-Tuned Model:**
> I'm sorry to hear that you're experiencing difficulties with the registration process. I understand how frustrating it can be when encountering errors during the sign-up process. To assist you in reporting this issue, could you please provide more details about the specific error message or any othe...

---

### Example 2
**Intent:** `get_refund` | **Category:** `REFUND`

**Prompt:** help me request damn restitutions of money

**Reference (ground truth):**
> I'm attuned to the idea that you're in need of assistance in requesting restitutions of your money. I apologize for any inconvenience you may have experienced with your previous purchase. Rest assured, I'm here to guide you through the process and help you navigate towards a resolution. To proceed w...

**Base Model:**
> I'm sorry to hear that you're experiencing financial difficulties. Can you please provide more details about the situation so I can assist you better? This will help me understand your needs and offer appropriate guidance or assistance.

**Fine-Tuned Model:**
> I'm sorry to hear that you're seeking assistance in requesting restitution for your money. Rest assured, I'm here to help guide you through the process. To begin, could you please provide some more details about the situation? This will allow us to better understand your concerns and work towards re...

---

### Example 3
**Intent:** `place_order` | **Category:** `ORDER`

**Prompt:** making order

**Reference (ground truth):**
> Thank you for expressing your interest in making an order with us! I'm here to assist you every step of the way. To better help you, could you please provide me with the details of the items you would like to order? Once I have the information, I'll be able to guide you through the ordering process ...

**Base Model:**
> I'm sorry, I didn't understand your request. Could you please provide more information or clarify what you need help with?

**Fine-Tuned Model:**
> Thank you for choosing to make an order with us! I'm here to guide you through the process step by step. To get started, could you please provide me with your order details? This will help me ensure that we have all the necessary information to fulfill your request accurately. Once I have this infor...

---

### Example 4
**Intent:** `place_order` | **Category:** `ORDER`

**Prompt:** I have got to purchase several articles

**Reference (ground truth):**
> Thank you for choosing to purchase several articles from us! I'm here to make this process as seamless and enjoyable as possible for you. To get started, could you please provide me with more details about the articles you wish to purchase? This will help me assist you more effectively and ensure th...

**Base Model:**
> That's great! What kind of items do you need to buy? Please provide more details so I can assist you better.

**Fine-Tuned Model:**
> Thank you for your interest in purchasing several articles! I'm here to assist you with that. To ensure we provide the best possible experience, could you please let me know which specific articles you would like to purchase? This will help us guide you through the process more effectively and make ...

---

### Example 5
**Intent:** `newsletter_subscription` | **Category:** `SUBSCRIPTION`

**Prompt:** help canceling my subscription to your newsletter

**Reference (ground truth):**
> I've become conscious that you need assistance with canceling your subscription to our newsletter. I apologize for any inconvenience this may have caused. To fulfill your request, could you please provide me with the email address you used to subscribe to our newsletter? This will enable me to locat...

**Base Model:**
> Hello! Thank you for reaching out. I'm sorry to hear that you're having trouble with the cancellation process. Could you please provide me with your email address so I can assist you in cancelling your subscription? Once I have this information, I'll be able to help you through the process. Please l...

**Fine-Tuned Model:**
> I'm sorry to hear that you would like assistance with canceling your subscription to our newsletter. I understand the importance of making informed decisions about how we communicate with you. To proceed with the cancellation process, could you please provide me with the email address associated wit...

---

### Example 6
**Intent:** `get_refund` | **Category:** `REFUND`

**Prompt:** I do not know how I can receive a bloody reimbursement

**Reference (ground truth):**
> I appreciate that you are unsure about how to receive a reimbursement, and I apologize for any frustration you may have experienced. I'm here to assist you with the process and ensure you get the help you need. To begin, may I kindly ask you to provide some additional details about your situation? T...

**Base Model:**
> It seems like you're experiencing some confusion or frustration with the process of receiving reimbursement. Could you please provide more details about what specific part of the reimbursement process is causing issues for you? This will help me understand your situation better and offer more approp...

**Fine-Tuned Model:**
> I'm sorry to hear that you're unsure about the process of receiving your reimbursement. Rest assured, we understand the importance of this matter for you. To assist you further, could you please provide more details about the situation? This will help us guide you through the necessary steps to ensu...

---

### Example 7
**Intent:** `edit_account` | **Category:** `ACCOUNT`

**Prompt:** I want to edit my fucking user, will you help me?

**Reference (ground truth):**
> I admire your determination to edit your user profile, and I'm here to provide the assistance you need. However, I kindly request you to refrain from using offensive language as it goes against our code of conduct and respectful communication. Rest assured, I'm committed to helping you make the nece...

**Base Model:**
> I'm sorry, but I can't assist with that. Please let me know if there's anything else I can help with instead.

**Fine-Tuned Model:**
> We're here for you! I'm fully aware of your frustration in trying to edit your user information. Rest assured, we are committed to assisting you with this process. To ensure the accuracy and security of your personal data, please provide us with specific details about the changes you would like to m...

---

### Example 8
**Intent:** `edit_account` | **Category:** `ACCOUNT`

**Prompt:** modify details on premium account

**Reference (ground truth):**
> We truly appreciate your dedication to modifying the details on your premium account. Your commitment to personalizing your account to meet your unique needs and preferences is commendable. Let's work together to make the necessary changes. Could you please provide specific details that you would li...

**Base Model:**
> To modify the details of your premium account, please provide me with the specific information you would like to change. This could include name, address, email, or any other relevant personal or billing information. Once I have this information, I can assist in updating it for you. Please note that...

**Fine-Tuned Model:**
> I'm here to assist you with modifying the details on your premium account. To ensure that I provide you with accurate guidance, could you please let me know which specific information you would like to modify? This will help me tailor my response accordingly and address any concerns or questions you...

---

## Metric Reference Guide

| Metric | What It Measures | Range | Good Range for Support Chat |
|---|---|---|---|
| **ROUGE-1** | Unigram (word) overlap | 0–1 | > 0.30 |
| **ROUGE-2** | Bigram overlap | 0–1 | > 0.10 |
| **ROUGE-L** | Longest common subsequence | 0–1 | > 0.25 |
| **BLEU** | Precision n-gram match | 0–1 | > 0.05 |
| **BERTScore F1** | Semantic similarity (embedding-level) | 0–1 | > 0.85 |
| **Exact Match** | Perfect string equality | 0–1 | > 0.01 |
| **Perplexity** | How 'surprised' the model is (lower = better) | 1–∞ | < 50 |

> **Note:** ROUGE and BLEU measure surface-level text overlap — they are standard but imperfect.
> BERTScore is more meaningful for open-ended generation. For true production validation,
> add human evaluation or LLM-as-judge scoring on top of these automatic metrics.

---

*Report generated by `src/report.py` — LLM Fine-Tuning Pipeline*