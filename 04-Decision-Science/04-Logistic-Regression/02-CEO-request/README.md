🏋️‍♀️ This is the final challenge of the week (you have until the end of the Communicate topic to complete it).

**Before diving into it, take time to finish the challenge 02 on sellers analysis (Liner Regression topic)**.

## Problem

>❓ How should Olist improve it's profit margin, given that it has:
> - some revenues per sellers per months
> - some revenues per orders
> - some reputation costs (estimated) per bad reviews
> - some operational costs of IT system that grows with number of orders, but not linearly (scale effects)

## Unit Economics (detailed)

***Revenue***

- Olist takes a **10% cut** on the product price (excl. freight) of each order delivered.
- Olist charges **80 BRL by month** per seller.

***Cost***

- In the long term, bad customer experience has business implications: low repeat rate, immediate customer support cost, refunds or unfavorable word of mouth communication. We will assume that we have an estimate measure of the monetary cost for each bad review:

review_score|cost (BRL)
---|---
1 star|100
2 stars|50
3 stars|40
4 stars|0
5 stars|0

In addition, Olist's **IT costs** (servers, etc...) increase with the amount of orders processed, albeit less and less rapidly (scale effects).
For the sake of simplicity, we will consider Olist's **total cumulated IT Costs** to be proportional to the **square-root** of the total cumulated number of orders approved.
The IT department also told you that since the birth of the marketplace, cumulated IT costs have amounted to 500,000 BRL.

## ✏️ Your turn!

👉 **Open the `ceo_request.ipynb` notebook and start from there.**

- We'll start from a blank Notebook
- Refrain from re-using from previous notebooks - they were made for investigation only
- All your re-usable logic is coded in `olist/*.py` scripts
